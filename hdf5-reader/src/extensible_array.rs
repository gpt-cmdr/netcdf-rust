//! HDF5 Extensible Array (EA) chunk index.
//!
//! This is the default chunk index for datasets with one unlimited dimension
//! and `libver='latest'`. It uses a three-level hierarchy:
//!
//! - `EAHD` — Extensible Array Header
//! - `EAIB` — Extensible Array Index Block
//! - `EADB` — Extensible Array Data Block
//! - `EASB` — Extensible Array Secondary Block

use crate::checksum::jenkins_lookup3;
use crate::chunk_index::{ChunkEntry, ChunkSet};
use crate::error::{Error, Result};
use crate::io::Cursor;
use crate::messages::dataspace::UNLIMITED;
use crate::storage::{Storage, StorageBuffer};

const EAHD_SIGNATURE: [u8; 4] = *b"EAHD";
const EAIB_SIGNATURE: [u8; 4] = *b"EAIB";
const EADB_SIGNATURE: [u8; 4] = *b"EADB";
const EASB_SIGNATURE: [u8; 4] = *b"EASB";

/// Parsed Extensible Array Header.
#[derive(Debug)]
struct EaHeader {
    client_id: u8,
    element_size: u8,
    max_nelmts_bits: u8,
    idx_blk_elmts: u8,
    data_blk_min_elmts: u8,
    sec_blk_min_data_ptrs: u8,
    max_dblk_page_nelmts_bits: u8,
    _nelmts: u64,
    index_block_address: u64,
}

/// Parse the Extensible Array Header.
///
/// On-disk layout (from H5EA_HEADER_SIZE):
/// sig(4) + ver(1) + client_id(1) +
/// element_size(1) + max_nelmts_bits(1) + idx_blk_elmts(1) +
/// data_blk_min_elmts(1) + sec_blk_min_data_ptrs(1) + max_dblk_page_nelmts_bits(1) +
/// 6 statistics fields (each length_size) +
/// index_block_address(offset_size) + checksum(4)
fn parse_header(data: &[u8], address: u64, offset_size: u8, length_size: u8) -> Result<EaHeader> {
    let mut cursor = Cursor::new(data);
    cursor.set_position(address);

    let sig = cursor.read_bytes(4)?;
    if sig != EAHD_SIGNATURE {
        return Err(Error::InvalidExtensibleArraySignature {
            context: "header signature mismatch",
        });
    }

    let version = cursor.read_u8()?;
    if version != 0 {
        return Err(Error::Other(format!(
            "unsupported extensible array header version {}",
            version
        )));
    }

    let client_id = cursor.read_u8()?;
    let element_size = cursor.read_u8()?;
    let max_nelmts_bits = cursor.read_u8()?;
    let idx_blk_elmts = cursor.read_u8()?;
    let data_blk_min_elmts = cursor.read_u8()?;
    let sec_blk_min_data_ptrs = cursor.read_u8()?;
    let max_dblk_page_nelmts_bits = cursor.read_u8()?;

    // Statistics (6 fields, each length_size bytes)
    let _nsuper_blks = cursor.read_length(length_size)?;
    let _super_blk_size = cursor.read_length(length_size)?;
    let _ndata_blks = cursor.read_length(length_size)?;
    let _data_blk_size = cursor.read_length(length_size)?;
    let _max_idx_set = cursor.read_length(length_size)?;
    let nelmts = cursor.read_length(length_size)?;

    let index_block_address = cursor.read_offset(offset_size)?;

    // Checksum
    let header_end = cursor.position();
    let header_bytes = &data[address as usize..header_end as usize];
    let stored_checksum = cursor.read_u32_le()?;
    let computed = jenkins_lookup3(header_bytes);
    if stored_checksum != computed {
        return Err(Error::ChecksumMismatch {
            expected: stored_checksum,
            actual: computed,
        });
    }

    Ok(EaHeader {
        client_id,
        element_size,
        max_nelmts_bits,
        idx_blk_elmts,
        data_blk_min_elmts,
        sec_blk_min_data_ptrs,
        max_dblk_page_nelmts_bits,
        _nelmts: nelmts,
        index_block_address,
    })
}

fn parse_header_storage(
    storage: &dyn Storage,
    address: u64,
    offset_size: u8,
    length_size: u8,
) -> Result<EaHeader> {
    let header_len = 4
        + 1
        + 1
        + 1
        + 1
        + 1
        + 1
        + 1
        + 1
        + 6 * usize::from(length_size)
        + usize::from(offset_size)
        + 4;
    let bytes = storage.read_range(address, header_len)?;
    parse_header(bytes.as_ref(), 0, offset_size, length_size)
}

/// Compute the super block layout, as libhdf5 does (`H5EA__hdr_init`).
///
/// Returns (elements_per_data_block, num_data_blocks) for each of the
/// `1 + max_nelmts_bits - log2(data_blk_min_elmts)` super blocks: super block
/// `u` holds `2^(u/2)` data blocks of `2^((u+1)/2) * data_blk_min_elmts`
/// elements each.
fn compute_super_block_layout(header: &EaHeader) -> Result<Vec<(u64, u64)>> {
    let invalid = || Error::InvalidData("invalid extensible array creation parameters".into());
    if !header.data_blk_min_elmts.is_power_of_two()
        || !header.sec_blk_min_data_ptrs.is_power_of_two()
        || header.max_nelmts_bits > 64
    {
        return Err(invalid());
    }
    let nsblks = u32::from(header.max_nelmts_bits)
        .checked_sub(header.data_blk_min_elmts.trailing_zeros())
        .ok_or_else(invalid)?
        + 1;

    Ok((0..nsblks)
        .map(|sb_idx| {
            (
                u64::from(header.data_blk_min_elmts) << sb_idx.div_ceil(2),
                1u64 << (sb_idx / 2),
            )
        })
        .collect())
}

/// The number of super blocks whose data blocks the index block points to
/// directly, and the number of those data blocks (`H5EA__iblock_alloc`).
fn index_block_direct_blocks(header: &EaHeader, sb_layout: &[(u64, u64)]) -> (usize, usize) {
    // `sec_blk_min_data_ptrs` is a power of two (checked by
    // `compute_super_block_layout`).
    let min_ptrs = header.sec_blk_min_data_ptrs;
    let super_blocks = (2 * min_ptrs.trailing_zeros() as usize).min(sb_layout.len());
    (super_blocks, 2 * (usize::from(min_ptrs) - 1))
}

fn checked_add_usize(left: usize, right: usize, context: &str) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| Error::InvalidData(format!("{context} exceeds platform usize capacity")))
}

fn checked_mul_usize(left: usize, right: usize, context: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| Error::InvalidData(format!("{context} exceeds platform usize capacity")))
}

fn checked_add_u64(left: u64, right: u64, context: &str) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| Error::InvalidData(format!("{context} exceeds u64 capacity")))
}

fn checked_usize_from_u64(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        Error::InvalidData(format!(
            "{context} value {value} exceeds platform usize capacity"
        ))
    })
}

fn checked_u64_from_usize(value: usize, context: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::InvalidData(format!("{context} exceeds u64 capacity")))
}

fn ea_block_header_len(offset_size: u8, sizeof_nelmts: usize) -> Result<usize> {
    let len = checked_add_usize(4, 1, "extensible array block header length")?;
    let len = checked_add_usize(len, 1, "extensible array block header length")?;
    let len = checked_add_usize(
        len,
        usize::from(offset_size),
        "extensible array block header length",
    )?;
    checked_add_usize(len, sizeof_nelmts, "extensible array block header length")
}

fn ea_page_nelmts(max_page_bits: u8) -> Result<usize> {
    if max_page_bits == 0 {
        return Ok(0);
    }
    1usize.checked_shl(u32::from(max_page_bits)).ok_or_else(|| {
        Error::InvalidData("extensible array page element count exceeds usize capacity".into())
    })
}

fn index_block_len(header: &EaHeader, sb_layout: &[(u64, u64)], offset_size: u8) -> Result<usize> {
    let offset_size = usize::from(offset_size);
    let inline_bytes = checked_mul_usize(
        usize::from(header.idx_blk_elmts),
        usize::from(header.element_size),
        "extensible array index block inline entry bytes",
    )?;
    let (direct_sblks, direct_ptrs) = index_block_direct_blocks(header, sb_layout);
    let direct_bytes = checked_mul_usize(
        direct_ptrs,
        offset_size,
        "extensible array index block direct data block address bytes",
    )?;
    let secondary_ptrs = sb_layout.len() - direct_sblks;
    let secondary_bytes = checked_mul_usize(
        secondary_ptrs,
        offset_size,
        "extensible array index block secondary block address bytes",
    )?;

    let mut len = 4usize;
    for part in [
        1,
        1,
        offset_size,
        inline_bytes,
        direct_bytes,
        secondary_bytes,
        4,
    ] {
        len = checked_add_usize(len, part, "extensible array index block length")?;
    }
    Ok(len)
}

/// Pages per data block of `elmts_per_dblk` elements, or 0 if such data
/// blocks are not paged (they fit in one page).
fn data_block_pages(page_nelmts: usize, elmts_per_dblk: u64) -> Result<usize> {
    if page_nelmts == 0 {
        return Ok(0);
    }
    let elmts_per_dblk =
        checked_usize_from_u64(elmts_per_dblk, "extensible array data block element count")?;
    if elmts_per_dblk <= page_nelmts {
        return Ok(0);
    }
    Ok(elmts_per_dblk.div_ceil(page_nelmts))
}

/// Size of a secondary block's page bitmap. libhdf5 reserves
/// `ceil(pages_per_dblk / 8)` bytes per data block, but numbers the bits
/// continuously: page `p` of data block `d` is bit `d * pages_per_dblk + p`.
fn secondary_page_bitmap_bytes(pages_per_dblk: usize, num_dblks: u64) -> Result<usize> {
    let num_dblks = checked_usize_from_u64(
        num_dblks,
        "extensible array secondary block data block count",
    )?;
    checked_mul_usize(
        num_dblks,
        pages_per_dblk.div_ceil(8),
        "extensible array secondary block page bitmap bytes",
    )
}

/// Whether bit `bit` of a page bitmap is set. libhdf5 numbers the bits from
/// the most significant bit of each byte (`H5VM_bit_get`).
fn page_initialized(page_init: &[u8], bit: usize) -> bool {
    page_init
        .get(bit / 8)
        .is_some_and(|byte| byte & (0x80 >> (bit % 8)) != 0)
}

fn secondary_block_len(
    num_dblk_addrs: usize,
    offset_size: u8,
    sizeof_nelmts: usize,
    page_bitmap_bytes: usize,
) -> Result<usize> {
    let offset_bytes = usize::from(offset_size);
    let addr_bytes = checked_mul_usize(
        num_dblk_addrs,
        offset_bytes,
        "extensible array secondary block data block address bytes",
    )?;

    let mut len = ea_block_header_len(offset_size, sizeof_nelmts)?;
    for part in [page_bitmap_bytes, addr_bytes, 4] {
        len = checked_add_usize(len, part, "extensible array secondary block length")?;
    }
    Ok(len)
}

/// A single raw entry.
struct EaRawEntry {
    address: u64,
    chunk_size: u64,
    filter_mask: u32,
}

/// Fields shared by every block of one array.
struct EaParams {
    is_filtered: bool,
    offset_size: u8,
    entry_size: u8,
    /// `ceil(max_nelmts_bits / 8)`, the width of the blocks' offset field.
    sizeof_nelmts: usize,
    /// Entries per data block page (0: data blocks are never paged).
    page_nelmts: usize,
}

/// A data block to read entries from.
struct DataBlockRef<'a> {
    address: u64,
    num_entries: u64,
    /// For data blocks reached through a secondary block: its page bitmap and
    /// the bit of this data block's first page.
    page_init: Option<(&'a [u8], usize)>,
}

/// Read `count` entries from the cursor.
fn read_entries(
    cursor: &mut Cursor<'_>,
    count: usize,
    is_filtered: bool,
    offset_size: u8,
    entry_size: u8,
) -> Result<Vec<EaRawEntry>> {
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let address = cursor.read_offset(offset_size)?;
        let (chunk_size, filter_mask) = if is_filtered {
            let chunk_size_len = entry_size
                .checked_sub(offset_size)
                .and_then(|remaining| remaining.checked_sub(4))
                .ok_or_else(|| Error::InvalidData("invalid extensible array entry size".into()))?;
            // 1–8 bytes wide (sized to the chunk), not a "length" of the superblock size.
            let cs = cursor.read_uvar(chunk_size_len as usize)?;
            let fm = cursor.read_u32_le()?;
            (cs, fm)
        } else {
            (0, 0)
        };
        entries.push(EaRawEntry {
            address,
            chunk_size,
            filter_mask,
        });
    }
    Ok(entries)
}

/// Read `count` entries at `position`.
fn read_entries_at_storage(
    storage: &dyn Storage,
    position: u64,
    count: usize,
    params: &EaParams,
) -> Result<Vec<EaRawEntry>> {
    let len = checked_mul_usize(
        count,
        usize::from(params.entry_size),
        "extensible array entry bytes",
    )?;
    let bytes = storage.read_range(position, len)?;
    let mut cursor = Cursor::new(bytes.as_ref());
    read_entries(
        &mut cursor,
        count,
        params.is_filtered,
        params.offset_size,
        params.entry_size,
    )
}

/// A parsed index block.
struct IndexBlock {
    /// Entries stored in the index block itself.
    inline: Vec<EaRawEntry>,
    /// Data block addresses of the first super blocks.
    dblk_addrs: Vec<u64>,
    /// Secondary block addresses of the remaining super blocks.
    sblk_addrs: Vec<u64>,
}

fn read_index_block_storage(
    storage: &dyn Storage,
    header: &EaHeader,
    sb_layout: &[(u64, u64)],
    params: &EaParams,
) -> Result<IndexBlock> {
    let offset_size = params.offset_size;
    let data = storage.read_range(
        header.index_block_address,
        index_block_len(header, sb_layout, offset_size)?,
    )?;
    let mut cursor = Cursor::new(data.as_ref());

    let sig = cursor.read_bytes(4)?;
    if sig != EAIB_SIGNATURE {
        return Err(Error::InvalidExtensibleArraySignature {
            context: "index block signature mismatch",
        });
    }

    let version = cursor.read_u8()?;
    if version != 0 {
        return Err(Error::Other(format!(
            "unsupported extensible array index block version {}",
            version
        )));
    }

    let _client_id = cursor.read_u8()?;
    let _header_address = cursor.read_offset(offset_size)?;

    let inline = read_entries(
        &mut cursor,
        usize::from(header.idx_blk_elmts),
        params.is_filtered,
        offset_size,
        header.element_size,
    )?;

    let (direct_sblks, ndblk_addrs) = index_block_direct_blocks(header, sb_layout);
    let mut dblk_addrs = Vec::with_capacity(ndblk_addrs);
    for _ in 0..ndblk_addrs {
        dblk_addrs.push(cursor.read_offset(offset_size)?);
    }

    let nsblk_addrs = sb_layout.len() - direct_sblks;
    let mut sblk_addrs = Vec::with_capacity(nsblk_addrs);
    for _ in 0..nsblk_addrs {
        sblk_addrs.push(cursor.read_offset(offset_size)?);
    }

    Ok(IndexBlock {
        inline,
        dblk_addrs,
        sblk_addrs,
    })
}

/// A parsed secondary block.
struct SecondaryBlock {
    dblk_addrs: Vec<u64>,
    /// Page bitmap of paged data blocks (see `secondary_page_bitmap_bytes`).
    page_init: Vec<u8>,
}

/// Parse a secondary block.
fn parse_secondary_block(
    data: &[u8],
    address: u64,
    num_dblk_addrs: usize,
    offset_size: u8,
    sizeof_nelmts: usize,
    page_bitmap_bytes: usize,
) -> Result<SecondaryBlock> {
    let mut cursor = Cursor::new(data);
    cursor.set_position(address);

    let sig = cursor.read_bytes(4)?;
    if sig != EASB_SIGNATURE {
        return Err(Error::InvalidExtensibleArraySignature {
            context: "secondary block signature mismatch",
        });
    }

    let version = cursor.read_u8()?;
    if version != 0 {
        return Err(Error::Other(format!(
            "unsupported extensible array secondary block version {}",
            version
        )));
    }

    let _client_id = cursor.read_u8()?;
    let _header_address = cursor.read_offset(offset_size)?;
    cursor.skip(sizeof_nelmts)?;

    // Per HDF5 spec III.H "Extensible Array Secondary Block", the secondary
    // block holds the page initialization bitmap when data blocks are paged.
    let page_init = cursor.read_bytes(page_bitmap_bytes)?.to_vec();

    let mut dblk_addrs = Vec::with_capacity(num_dblk_addrs);
    for _ in 0..num_dblk_addrs {
        dblk_addrs.push(cursor.read_offset(offset_size)?);
    }

    // Skip checksum
    let _checksum = cursor.read_u32_le()?;

    Ok(SecondaryBlock {
        dblk_addrs,
        page_init,
    })
}

fn parse_secondary_block_storage(
    storage: &dyn Storage,
    address: u64,
    elmts_per_dblk: u64,
    num_dblks: u64,
    params: &EaParams,
) -> Result<SecondaryBlock> {
    let num_dblk_addrs = checked_usize_from_u64(
        num_dblks,
        "extensible array secondary block data block count",
    )?;
    let page_bitmap_bytes = secondary_page_bitmap_bytes(
        data_block_pages(params.page_nelmts, elmts_per_dblk)?,
        num_dblks,
    )?;
    let read_len = secondary_block_len(
        num_dblk_addrs,
        params.offset_size,
        params.sizeof_nelmts,
        page_bitmap_bytes,
    )?;
    let bytes = storage.read_range(address, read_len)?;
    parse_secondary_block(
        bytes.as_ref(),
        0,
        num_dblk_addrs,
        params.offset_size,
        params.sizeof_nelmts,
        page_bitmap_bytes,
    )
}

/// Check a data block's signature and version; returns the length of its
/// prefix (signature, version, client ID, header address and block offset).
fn check_data_block_prefix_storage(
    storage: &dyn Storage,
    address: u64,
    params: &EaParams,
) -> Result<usize> {
    let prefix_len = ea_block_header_len(params.offset_size, params.sizeof_nelmts)?;
    let prefix = storage.read_range(address, prefix_len)?;
    let mut cursor = Cursor::new(prefix.as_ref());

    let sig = cursor.read_bytes(4)?;
    if sig != EADB_SIGNATURE {
        return Err(Error::InvalidExtensibleArraySignature {
            context: "data block signature mismatch",
        });
    }

    let version = cursor.read_u8()?;
    if version != 0 {
        return Err(Error::Other(format!(
            "unsupported extensible array data block version {}",
            version
        )));
    }

    Ok(prefix_len)
}

/// File position of the first entry of page `page_idx` of a paged data block.
///
/// A data block of more than one page is its prefix and the prefix checksum,
/// followed by every page at a fixed stride (page entries and a checksum),
/// whether or not the page was ever written (`H5EA__lookup_elmt`).
fn data_block_page_position(
    address: u64,
    prefix_len: usize,
    page_idx: usize,
    params: &EaParams,
) -> Result<u64> {
    let page_bytes = checked_mul_usize(
        params.page_nelmts,
        usize::from(params.entry_size),
        "extensible array data block page entry bytes",
    )?;
    let page_size = checked_add_usize(page_bytes, 4, "extensible array data block page size")?;
    let pages_offset = checked_mul_usize(
        page_idx,
        page_size,
        "extensible array data block page offset",
    )?;
    let offset = checked_add_usize(prefix_len, 4, "extensible array data block page offset")?;
    let offset = checked_add_usize(
        offset,
        pages_offset,
        "extensible array data block page offset",
    )?;
    checked_add_u64(
        address,
        checked_u64_from_usize(offset, "extensible array data block page offset")?,
        "extensible array data block page position",
    )
}

/// Whether page `page_idx` of a paged data block was ever written.
fn data_block_page_initialized(block: &DataBlockRef<'_>, page_idx: usize) -> Result<bool> {
    let Some((page_init, first_bit)) = block.page_init else {
        return Err(Error::InvalidData(
            "paged extensible array data block outside a secondary block".into(),
        ));
    };
    let bit = checked_add_usize(first_bit, page_idx, "extensible array page bitmap index")?;
    Ok(page_initialized(page_init, bit))
}

/// Read entry `local_idx` of a data block, or `None` if it lies in a page that
/// was never written.
fn read_data_block_entry_storage(
    storage: &dyn Storage,
    block: &DataBlockRef<'_>,
    local_idx: usize,
    params: &EaParams,
) -> Result<Option<EaRawEntry>> {
    let prefix_len = check_data_block_prefix_storage(storage, block.address, params)?;
    let entry_size = usize::from(params.entry_size);

    let position = if data_block_pages(params.page_nelmts, block.num_entries)? == 0 {
        let offset = checked_mul_usize(local_idx, entry_size, "EA entry offset")?;
        let offset = checked_add_usize(prefix_len, offset, "EA entry offset")?;
        checked_add_u64(
            block.address,
            checked_u64_from_usize(offset, "EA entry offset")?,
            "EA entry position",
        )?
    } else {
        let page_idx = local_idx / params.page_nelmts;
        if !data_block_page_initialized(block, page_idx)? {
            return Ok(None);
        }
        let within_page = checked_mul_usize(
            local_idx % params.page_nelmts,
            entry_size,
            "EA page entry offset",
        )?;
        checked_add_u64(
            data_block_page_position(block.address, prefix_len, page_idx, params)?,
            checked_u64_from_usize(within_page, "EA page entry offset")?,
            "EA page entry position",
        )?
    };

    Ok(read_entries_at_storage(storage, position, 1, params)?.pop())
}

/// Read entries `0..count` of a data block, each with its index in the block.
/// Entries in pages that were never written are left out.
fn read_data_block_storage(
    storage: &dyn Storage,
    block: &DataBlockRef<'_>,
    count: usize,
    params: &EaParams,
) -> Result<Vec<(usize, EaRawEntry)>> {
    let prefix_len = check_data_block_prefix_storage(storage, block.address, params)?;

    if data_block_pages(params.page_nelmts, block.num_entries)? == 0 {
        let position = checked_add_u64(
            block.address,
            checked_u64_from_usize(prefix_len, "EA data block prefix length")?,
            "EA data block entry position",
        )?;
        let entries = read_entries_at_storage(storage, position, count, params)?;
        return Ok(entries.into_iter().enumerate().collect());
    }

    let mut entries = Vec::new();
    for page_idx in 0..count.div_ceil(params.page_nelmts) {
        if !data_block_page_initialized(block, page_idx)? {
            continue;
        }
        let first = page_idx * params.page_nelmts;
        let page_entries = read_entries_at_storage(
            storage,
            data_block_page_position(block.address, prefix_len, page_idx, params)?,
            params.page_nelmts.min(count - first),
            params,
        )?;
        entries.extend(
            page_entries
                .into_iter()
                .enumerate()
                .map(|(i, entry)| (first + i, entry)),
        );
    }
    Ok(entries)
}

/// How the chunk index numbers a dataset's chunks (`H5D__earray_idx_get_addr`):
/// row-major over the chunk counts at the maximum dimensions, with the
/// unlimited dimension moved to the front.
struct ChunkNumbering {
    /// Dimensions in numbering order: the unlimited one, then the others.
    order: Vec<usize>,
    /// Weight of each position of `order` in the element index.
    weights: Vec<u64>,
    /// Chunk count at the maximum size, per position of `order`.
    max_counts: Vec<u64>,
    /// Current chunk count per dimension.
    counts: Vec<u64>,
}

impl ChunkNumbering {
    fn new(dataset_shape: &[u64], chunk_dims: &[u32], max_dims: Option<&[u64]>) -> Self {
        let ndim = dataset_shape.len();
        let counts: Vec<u64> = (0..ndim)
            .map(|dim| dataset_shape[dim].div_ceil(u64::from(chunk_dims[dim])))
            .collect();
        let max_dims = max_dims.filter(|max_dims| max_dims.len() == ndim);
        let unlimited = max_dims
            .and_then(|max_dims| max_dims.iter().position(|&dim| dim == UNLIMITED))
            .unwrap_or(0);
        let order: Vec<usize> = std::iter::once(unlimited)
            .chain((0..ndim).filter(|&dim| dim != unlimited))
            .take(ndim)
            .collect();
        let max_counts: Vec<u64> = order
            .iter()
            .map(|&dim| match max_dims {
                Some(max_dims) if max_dims[dim] != UNLIMITED => max_dims[dim]
                    .div_ceil(u64::from(chunk_dims[dim]))
                    .max(counts[dim]),
                _ => counts[dim],
            })
            .collect();
        let mut weights = vec![1u64; ndim];
        for pos in (0..ndim.saturating_sub(1)).rev() {
            weights[pos] = weights[pos + 1].saturating_mul(max_counts[pos + 1]);
        }
        ChunkNumbering {
            order,
            weights,
            max_counts,
            counts,
        }
    }

    /// Element index of the chunk at `chunk_indices` (per dimension).
    fn index(&self, chunk_indices: &[u64]) -> u64 {
        self.order
            .iter()
            .zip(&self.weights)
            .fold(0u64, |index, (&dim, &weight)| {
                index.saturating_add(chunk_indices[dim].saturating_mul(weight))
            })
    }

    /// Chunk indices (per dimension) of element `index`, or `None` if that
    /// chunk lies outside the dataset's current extent.
    fn chunk_indices(&self, index: u64) -> Option<Vec<u64>> {
        let mut chunk_indices = vec![0u64; self.order.len()];
        for (pos, &dim) in self.order.iter().enumerate() {
            let mut chunk_index = index / self.weights[pos];
            if pos > 0 {
                chunk_index %= self.max_counts[pos];
            }
            if chunk_index >= self.counts[dim] {
                return None;
            }
            chunk_indices[dim] = chunk_index;
        }
        Some(chunk_indices)
    }

    /// One past the largest element index of a chunk inside the extent.
    fn end(&self) -> u64 {
        if self.counts.contains(&0) {
            return 0;
        }
        let last: Vec<u64> = self.counts.iter().map(|count| count - 1).collect();
        self.index(&last).saturating_add(1)
    }
}

fn linear_target_offsets(
    numbering: &ChunkNumbering,
    chunk_dims: &[u32],
    chunk_bounds: (&[u64], &[u64]),
) -> Vec<(u64, Vec<u64>)> {
    let ndim = numbering.counts.len();
    if ndim == 0 {
        return vec![(0, Vec::new())];
    }

    let (first_chunk, last_chunk) = chunk_bounds;
    let mut targets = Vec::new();
    let mut chunk_indices = first_chunk.to_vec();
    loop {
        let offsets = chunk_indices
            .iter()
            .enumerate()
            .map(|(dim, chunk_index)| chunk_index * u64::from(chunk_dims[dim]))
            .collect();
        targets.push((numbering.index(&chunk_indices), offsets));

        let mut advanced = false;
        for dim in (0..ndim).rev() {
            if chunk_indices[dim] < last_chunk[dim] {
                chunk_indices[dim] += 1;
                if dim + 1 < ndim {
                    chunk_indices[(dim + 1)..ndim].copy_from_slice(&first_chunk[(dim + 1)..ndim]);
                }
                advanced = true;
                break;
            }
        }

        if !advanced {
            break;
        }
    }

    targets
}

/// Look up the entries at `targets` (element index, chunk offsets).
fn collect_extensible_array_chunk_entries_at_storage(
    storage: &dyn Storage,
    header: &EaHeader,
    sb_layout: &[(u64, u64)],
    params: &EaParams,
    targets: Vec<(u64, Vec<u64>)>,
) -> Result<Vec<ChunkEntry>> {
    let offset_size = params.offset_size;
    let index = read_index_block_storage(storage, header, sb_layout, params)?;
    let (direct_sblks, _) = index_block_direct_blocks(header, sb_layout);

    let mut secondary_block_cache: Vec<Option<SecondaryBlock>> =
        (0..index.sblk_addrs.len()).map(|_| None).collect();
    let mut entries = Vec::new();

    for (linear_idx, offsets) in targets {
        let looked_up;
        let inline = usize::try_from(linear_idx)
            .ok()
            .and_then(|idx| index.inline.get(idx));
        let raw = if let Some(raw) = inline {
            raw
        } else {
            // Element index past the index block's own entries.
            let mut relative_idx = linear_idx - u64::from(header.idx_blk_elmts);
            let mut sb_idx = None;
            for (candidate_idx, (elmts_per_dblk, num_dblks)) in sb_layout.iter().enumerate() {
                let capacity = elmts_per_dblk.saturating_mul(*num_dblks);
                if relative_idx < capacity {
                    sb_idx = Some(candidate_idx);
                    break;
                }
                relative_idx -= capacity;
            }

            let Some(sb_idx) = sb_idx else {
                continue;
            };
            let (elmts_per_dblk, num_dblks) = sb_layout[sb_idx];
            let dblk_idx = checked_usize_from_u64(
                relative_idx / elmts_per_dblk,
                "extensible array data block index",
            )?;
            let local_idx = checked_usize_from_u64(
                relative_idx % elmts_per_dblk,
                "extensible array data block element index",
            )?;

            let (dblk_addr, page_init) = if sb_idx < direct_sblks {
                // Super blocks before `sb_idx` are direct too, with few data blocks.
                let base = sb_layout[..sb_idx]
                    .iter()
                    .map(|(_, num_dblks)| *num_dblks as usize)
                    .sum::<usize>();
                let dblk_addr = index
                    .dblk_addrs
                    .get(base + dblk_idx)
                    .copied()
                    .unwrap_or(u64::MAX);
                (dblk_addr, None)
            } else {
                let sec_cache_idx = sb_idx - direct_sblks;
                if secondary_block_cache[sec_cache_idx].is_none() {
                    let sec_addr = index
                        .sblk_addrs
                        .get(sec_cache_idx)
                        .copied()
                        .unwrap_or(u64::MAX);
                    secondary_block_cache[sec_cache_idx] =
                        Some(if Cursor::is_undefined_offset(sec_addr, offset_size) {
                            SecondaryBlock {
                                dblk_addrs: Vec::new(),
                                page_init: Vec::new(),
                            }
                        } else {
                            parse_secondary_block_storage(
                                storage,
                                sec_addr,
                                elmts_per_dblk,
                                num_dblks,
                                params,
                            )?
                        });
                }

                let Some(secondary) = secondary_block_cache[sec_cache_idx].as_ref() else {
                    continue;
                };
                let dblk_addr = secondary
                    .dblk_addrs
                    .get(dblk_idx)
                    .copied()
                    .unwrap_or(u64::MAX);
                let first_page_bit = checked_mul_usize(
                    dblk_idx,
                    data_block_pages(params.page_nelmts, elmts_per_dblk)?,
                    "extensible array page bitmap index",
                )?;
                (
                    dblk_addr,
                    Some((secondary.page_init.as_slice(), first_page_bit)),
                )
            };

            if Cursor::is_undefined_offset(dblk_addr, offset_size) {
                continue;
            }

            let block = DataBlockRef {
                address: dblk_addr,
                num_entries: elmts_per_dblk,
                page_init,
            };
            let Some(raw) = read_data_block_entry_storage(storage, &block, local_idx, params)?
            else {
                continue;
            };
            looked_up = raw;
            &looked_up
        };

        if Cursor::is_undefined_offset(raw.address, offset_size) {
            continue;
        }

        entries.push(ChunkEntry {
            address: raw.address,
            size: raw.chunk_size,
            filter_mask: raw.filter_mask,
            offsets,
        });
    }

    Ok(entries)
}

/// In-memory bytes viewed as storage.
struct SliceStorage<'a>(&'a [u8]);

impl Storage for SliceStorage<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_range(&self, offset: u64, len: usize) -> Result<StorageBuffer> {
        let start = usize::try_from(offset).map_err(|_| Error::OffsetOutOfBounds(offset))?;
        let end = start
            .checked_add(len)
            .ok_or(Error::OffsetOutOfBounds(offset))?;
        if end > self.0.len() {
            return Err(Error::UnexpectedEof {
                offset,
                needed: len as u64,
                available: self.len().saturating_sub(offset),
            });
        }
        Ok(StorageBuffer::from_vec(self.0[start..end].to_vec()))
    }
}

/// Collect chunk entries from an Extensible Array index.
///
/// Walks the EAHD → EAIB → (EADB / EASB → EADB) hierarchy and converts
/// linear entry indices to multi-dimensional chunk offsets. `max_dims` are
/// the dataspace's maximum dimensions: libhdf5 numbers the chunks over them,
/// with the unlimited dimension first. Without them the current dimensions
/// are used, which is right only for a first unlimited dimension whose other
/// dimensions are at their maximum.
#[allow(clippy::too_many_arguments)]
pub fn collect_extensible_array_chunk_entries(
    data: &[u8],
    header_address: u64,
    offset_size: u8,
    length_size: u8,
    dataset_shape: &[u64],
    max_dims: Option<&[u64]>,
    chunk_dims: &[u32],
    chunk_bounds: Option<(&[u64], &[u64])>,
) -> Result<Vec<ChunkEntry>> {
    collect_extensible_array_chunk_entries_storage(
        &SliceStorage(data),
        header_address,
        offset_size,
        length_size,
        dataset_shape,
        max_dims,
        chunk_dims,
        chunk_bounds,
    )
}

/// Collect the entries of the chunks in `set` from an Extensible Array
/// index, reading only the blocks that hold their entries.
#[allow(clippy::too_many_arguments)]
pub(crate) fn collect_extensible_array_chunk_entries_in_set_storage(
    storage: &dyn Storage,
    header_address: u64,
    offset_size: u8,
    length_size: u8,
    dataset_shape: &[u64],
    max_dims: Option<&[u64]>,
    chunk_dims: &[u32],
    set: &ChunkSet,
) -> Result<Vec<ChunkEntry>> {
    let header = parse_header_storage(storage, header_address, offset_size, length_size)?;
    if Cursor::is_undefined_offset(header.index_block_address, offset_size)
        || set.rank() != dataset_shape.len()
    {
        return Ok(Vec::new());
    }
    let sb_layout = compute_super_block_layout(&header)?;
    let params = EaParams {
        is_filtered: header.client_id == 1,
        offset_size,
        entry_size: header.element_size,
        sizeof_nelmts: usize::from(header.max_nelmts_bits).div_ceil(8),
        page_nelmts: ea_page_nelmts(header.max_dblk_page_nelmts_bits)?,
    };
    let numbering = ChunkNumbering::new(dataset_shape, chunk_dims, max_dims);
    let targets = set
        .chunks()
        .filter(|chunk_indices| {
            chunk_indices
                .iter()
                .zip(&numbering.counts)
                .all(|(index, count)| index < count)
        })
        .map(|chunk_indices| {
            let offsets = chunk_indices
                .iter()
                .zip(chunk_dims)
                .map(|(&index, &chunk)| index * u64::from(chunk))
                .collect();
            (numbering.index(&chunk_indices), offsets)
        })
        .collect();
    collect_extensible_array_chunk_entries_at_storage(
        storage, &header, &sb_layout, &params, targets,
    )
}

/// Collect chunk entries from an Extensible Array index using random-access
/// storage. See [`collect_extensible_array_chunk_entries`].
#[allow(clippy::too_many_arguments)]
pub fn collect_extensible_array_chunk_entries_storage(
    storage: &dyn Storage,
    header_address: u64,
    offset_size: u8,
    length_size: u8,
    dataset_shape: &[u64],
    max_dims: Option<&[u64]>,
    chunk_dims: &[u32],
    chunk_bounds: Option<(&[u64], &[u64])>,
) -> Result<Vec<ChunkEntry>> {
    let header = parse_header_storage(storage, header_address, offset_size, length_size)?;

    if Cursor::is_undefined_offset(header.index_block_address, offset_size) {
        return Ok(Vec::new());
    }

    let sb_layout = compute_super_block_layout(&header)?;
    let params = EaParams {
        is_filtered: header.client_id == 1,
        offset_size,
        entry_size: header.element_size,
        sizeof_nelmts: usize::from(header.max_nelmts_bits).div_ceil(8),
        page_nelmts: ea_page_nelmts(header.max_dblk_page_nelmts_bits)?,
    };

    let numbering = ChunkNumbering::new(dataset_shape, chunk_dims, max_dims);
    if let Some(bounds) = chunk_bounds {
        let targets = linear_target_offsets(&numbering, chunk_dims, bounds);
        return collect_extensible_array_chunk_entries_at_storage(
            storage, &header, &sb_layout, &params, targets,
        );
    }

    // Entries past the last chunk inside the extent are not read.
    let num_chunks = numbering.end();

    let index = read_index_block_storage(storage, &header, &sb_layout, &params)?;
    let (direct_sblks, _) = index_block_direct_blocks(&header, &sb_layout);

    // (element index, entry) of every entry read, in element order.
    let mut all_entries: Vec<(u64, EaRawEntry)> = index
        .inline
        .into_iter()
        .enumerate()
        .map(|(idx, raw)| (idx as u64, raw))
        .collect();

    // Element index of the next data block's first entry.
    let mut dblk_start = u64::from(header.idx_blk_elmts);
    let read_data_block = |block: DataBlockRef<'_>,
                           dblk_start: u64,
                           all_entries: &mut Vec<(u64, EaRawEntry)>|
     -> Result<()> {
        if Cursor::is_undefined_offset(block.address, offset_size) {
            return Ok(());
        }
        let count = checked_usize_from_u64(
            block.num_entries.min(num_chunks - dblk_start),
            "extensible array data block element count",
        )?;
        for (local_idx, raw) in read_data_block_storage(storage, &block, count, &params)? {
            all_entries.push((dblk_start + local_idx as u64, raw));
        }
        Ok(())
    };

    let mut direct_dblk_addrs = index.dblk_addrs.iter().copied();
    for (sb_idx, &(elmts_per_dblk, num_dblks)) in sb_layout.iter().enumerate() {
        if dblk_start >= num_chunks {
            break;
        }

        if sb_idx < direct_sblks {
            for _ in 0..num_dblks {
                if dblk_start >= num_chunks {
                    break;
                }
                let block = DataBlockRef {
                    address: direct_dblk_addrs.next().unwrap_or(u64::MAX),
                    num_entries: elmts_per_dblk,
                    page_init: None,
                };
                read_data_block(block, dblk_start, &mut all_entries)?;
                dblk_start = dblk_start.saturating_add(elmts_per_dblk);
            }
            continue;
        }

        let sec_addr = index
            .sblk_addrs
            .get(sb_idx - direct_sblks)
            .copied()
            .unwrap_or(u64::MAX);
        if Cursor::is_undefined_offset(sec_addr, offset_size) {
            dblk_start = dblk_start.saturating_add(elmts_per_dblk.saturating_mul(num_dblks));
            continue;
        }

        let secondary =
            parse_secondary_block_storage(storage, sec_addr, elmts_per_dblk, num_dblks, &params)?;
        let pages_per_dblk = data_block_pages(params.page_nelmts, elmts_per_dblk)?;
        for (dblk_idx, &dblk_addr) in secondary.dblk_addrs.iter().enumerate() {
            if dblk_start >= num_chunks {
                break;
            }
            let block = DataBlockRef {
                address: dblk_addr,
                num_entries: elmts_per_dblk,
                page_init: Some((
                    secondary.page_init.as_slice(),
                    checked_mul_usize(
                        dblk_idx,
                        pages_per_dblk,
                        "extensible array page bitmap index",
                    )?,
                )),
            };
            read_data_block(block, dblk_start, &mut all_entries)?;
            dblk_start = dblk_start.saturating_add(elmts_per_dblk);
        }
    }

    let mut entries = Vec::new();
    for (linear_idx, raw) in all_entries {
        if linear_idx >= num_chunks || Cursor::is_undefined_offset(raw.address, offset_size) {
            continue;
        }
        // Entries of chunks outside the extent (along a dimension below its
        // maximum) are skipped.
        let Some(chunk_indices) = numbering.chunk_indices(linear_idx) else {
            continue;
        };
        let offsets = chunk_indices
            .iter()
            .enumerate()
            .map(|(dim, chunk_index)| chunk_index * u64::from(chunk_dims[dim]))
            .collect();

        entries.push(ChunkEntry {
            address: raw.address,
            size: raw.chunk_size,
            filter_mask: raw.filter_mask,
            offsets,
        });
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{Storage, StorageBuffer};
    use std::sync::Mutex;

    const TEST_OFFSET_SIZE: u8 = 8;
    const TEST_LENGTH_SIZE: u8 = 8;
    const TEST_HEADER_ADDR: u64 = 0;
    const TEST_INDEX_ADDR: u64 = 128;
    const TEST_DIRECT_DATA_BLOCK_ADDR: u64 = 256;
    const TEST_SECONDARY_ADDR: u64 = 512;
    const TEST_DATA_BLOCK_ADDR: u64 = 576;
    const TEST_PAGED_SECONDARY_ADDR: u64 = 640;
    const TEST_PAGED_DATA_BLOCK_ADDRS: [u64; 2] = [768, 896];
    /// Elements of the test array that are set; element `i` points at
    /// `0x1000 + i`. See `storage_fixture` for the layout.
    const TEST_SET_ELEMENTS: [u64; 15] = [0, 1, 2, 7, 8, 9, 10, 19, 20, 21, 22, 23, 24, 25, 26];

    struct RecordingStorage {
        data: Vec<u8>,
        ranges: Mutex<Vec<(u64, usize)>>,
    }

    impl RecordingStorage {
        fn new(data: Vec<u8>) -> Self {
            Self {
                data,
                ranges: Mutex::new(Vec::new()),
            }
        }

        fn ranges(&self) -> Vec<(u64, usize)> {
            self.ranges.lock().unwrap().clone()
        }

        fn clear_ranges(&self) {
            self.ranges.lock().unwrap().clear();
        }
    }

    impl Storage for RecordingStorage {
        fn len(&self) -> u64 {
            self.data.len() as u64
        }

        fn read_range(&self, offset: u64, len: usize) -> Result<StorageBuffer> {
            let start = usize::try_from(offset).map_err(|_| Error::OffsetOutOfBounds(offset))?;
            let end = start.checked_add(len).ok_or(Error::UnexpectedEof {
                offset,
                needed: len as u64,
                available: self.data.len().saturating_sub(start) as u64,
            })?;
            if end > self.data.len() {
                return Err(Error::UnexpectedEof {
                    offset,
                    needed: len as u64,
                    available: self.data.len().saturating_sub(start) as u64,
                });
            }
            self.ranges.lock().unwrap().push((offset, len));
            Ok(StorageBuffer::from_vec(self.data[start..end].to_vec()))
        }
    }

    fn put(data: &mut [u8], offset: u64, bytes: &[u8]) {
        let start = offset as usize;
        data[start..start + bytes.len()].copy_from_slice(bytes);
    }

    fn push_u64(buf: &mut Vec<u8>, value: u64) {
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn push_offset(buf: &mut Vec<u8>, value: u64) {
        push_u64(buf, value);
    }

    fn block_prefix(signature: [u8; 4]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&signature);
        buf.push(0); // version
        buf.push(0); // client id
        push_offset(&mut buf, TEST_HEADER_ADDR);
        buf
    }

    /// An unfiltered array of 8-byte addresses with the creation parameters
    /// below. Its super blocks (u: data blocks x elements) are 0: 1 x 2,
    /// 1: 1 x 4, 2: 2 x 4, 3: 2 x 8, ...; the index block holds element 0 and
    /// points directly to the data blocks of super blocks 0 and 1; super
    /// blocks 3 and later have data blocks of two 4-element pages.
    fn ea_header_bytes(nelmts: u64) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&EAHD_SIGNATURE);
        buf.push(0); // version
        buf.push(0); // unfiltered raw chunk client
        buf.push(8); // element size: address only
        buf.push(8); // max_nelmts_bits, so sizeof_nelmts is 1 and there are 8 super blocks
        buf.push(1); // inline elements
        buf.push(2); // minimum data block elements
        buf.push(2); // minimum secondary block data pointers
        buf.push(2); // data block pages of 4 elements
        for value in [0, 0, 0, 0, 0, nelmts] {
            push_u64(&mut buf, value);
        }
        push_offset(&mut buf, TEST_INDEX_ADDR);
        let checksum = jenkins_lookup3(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        buf
    }

    fn index_block_bytes() -> Vec<u8> {
        let mut buf = block_prefix(EAIB_SIGNATURE);
        push_offset(&mut buf, 0x1000); // element 0
        push_offset(&mut buf, TEST_DIRECT_DATA_BLOCK_ADDR); // super block 0: elements 1-2
        push_offset(&mut buf, u64::MAX); // super block 1: elements 3-6, unset
        push_offset(&mut buf, TEST_SECONDARY_ADDR); // super block 2: elements 7-14
        push_offset(&mut buf, TEST_PAGED_SECONDARY_ADDR); // super block 3: elements 15-30
        for _ in 4..8 {
            push_offset(&mut buf, u64::MAX); // super blocks 4-7
        }
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }

    fn data_block_bytes(entries: &[u64]) -> Vec<u8> {
        let mut buf = block_prefix(EADB_SIGNATURE);
        buf.push(0); // block offset, sizeof_nelmts = 1
        for &entry in entries {
            push_offset(&mut buf, entry);
        }
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }

    fn secondary_block_bytes(page_init: &[u8], dblk_addrs: &[u64]) -> Vec<u8> {
        let mut buf = block_prefix(EASB_SIGNATURE);
        buf.push(0); // block offset, sizeof_nelmts = 1
        buf.extend_from_slice(page_init);
        for &addr in dblk_addrs {
            push_offset(&mut buf, addr);
        }
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }

    /// A paged data block: prefix, prefix checksum, then two pages of four
    /// entries and a checksum each. Unwritten pages are filled with garbage.
    fn paged_data_block_bytes(pages: [Option<u64>; 2]) -> Vec<u8> {
        let mut buf = block_prefix(EADB_SIGNATURE);
        buf.push(0); // block offset, sizeof_nelmts = 1
        buf.extend_from_slice(&0u32.to_le_bytes());
        for page in pages {
            for i in 0..4 {
                push_offset(&mut buf, page.map_or(0xdead, |first| first + i));
            }
            buf.extend_from_slice(&0u32.to_le_bytes());
        }
        buf
    }

    fn storage_fixture() -> RecordingStorage {
        let mut data = vec![0u8; 4096];
        put(&mut data, TEST_HEADER_ADDR, &ea_header_bytes(31));
        put(&mut data, TEST_INDEX_ADDR, &index_block_bytes());
        put(
            &mut data,
            TEST_DIRECT_DATA_BLOCK_ADDR,
            &data_block_bytes(&[0x1001, 0x1002]),
        );
        // Super block 2: elements 7-10 set, 11-14 in an unset data block.
        put(
            &mut data,
            TEST_SECONDARY_ADDR,
            &secondary_block_bytes(&[], &[TEST_DATA_BLOCK_ADDR, u64::MAX]),
        );
        put(
            &mut data,
            TEST_DATA_BLOCK_ADDR,
            &data_block_bytes(&[0x1007, 0x1008, 0x1009, 0x100a]),
        );
        // Super block 3: page bits 0-1 belong to the first data block (elements
        // 15-22), bits 2-3 to the second (23-30). Only the first data block's
        // second page (19-22) and the second data block's first page (23-26)
        // were written: bits 1 and 2, most significant bit first.
        put(
            &mut data,
            TEST_PAGED_SECONDARY_ADDR,
            &secondary_block_bytes(&[0b0110_0000, 0], &TEST_PAGED_DATA_BLOCK_ADDRS),
        );
        put(
            &mut data,
            TEST_PAGED_DATA_BLOCK_ADDRS[0],
            &paged_data_block_bytes([None, Some(0x1013)]),
        );
        put(
            &mut data,
            TEST_PAGED_DATA_BLOCK_ADDRS[1],
            &paged_data_block_bytes([Some(0x1017), None]),
        );
        RecordingStorage::new(data)
    }

    fn entry_addresses(entries: &[ChunkEntry]) -> Vec<(u64, u64)> {
        entries
            .iter()
            .map(|entry| (entry.offsets[0], entry.address))
            .collect()
    }

    fn expected_addresses() -> Vec<(u64, u64)> {
        TEST_SET_ELEMENTS
            .iter()
            .map(|&idx| (idx, 0x1000 + idx))
            .collect()
    }

    #[test]
    fn chunk_numbering_follows_libhdf5() {
        const U: u64 = UNLIMITED;
        // First unlimited dimension, the other at its maximum: row-major.
        let n = ChunkNumbering::new(&[5, 8], &[2, 4], Some(&[U, 8]));
        assert_eq!(n.index(&[2, 1]), 5);
        assert_eq!(n.end(), 6);
        // A larger maximum widens every row: 4 chunks per row, 2 in the extent.
        let n = ChunkNumbering::new(&[5, 8], &[2, 4], Some(&[U, 16]));
        assert_eq!(n.index(&[1, 1]), 5);
        assert_eq!(n.chunk_indices(5), Some(vec![1, 1]));
        assert_eq!(n.chunk_indices(6), None);
        assert_eq!(n.end(), 10);
        // Unlimited last: numbered as if it were first.
        let n = ChunkNumbering::new(&[8, 5], &[4, 2], Some(&[8, U]));
        assert_eq!(n.index(&[1, 0]), 1);
        assert_eq!(n.index(&[0, 1]), 2);
        assert_eq!(n.chunk_indices(5), Some(vec![1, 2]));
        // Unlimited in the middle of three dimensions.
        let n = ChunkNumbering::new(&[2, 25, 3], &[1, 1, 3], Some(&[2, U, 3]));
        assert_eq!(n.index(&[1, 3, 0]), 3 * 2 + 1);
        assert_eq!(n.end(), 50);
        // Without maximum dimensions: the current ones, row-major.
        let n = ChunkNumbering::new(&[5, 8], &[2, 4], None);
        assert_eq!(n.index(&[2, 1]), 5);
    }

    #[test]
    fn eahd_bad_signature() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"XXXX");
        let err = parse_header(&data, 0, 8, 8).unwrap_err();
        assert!(matches!(err, Error::InvalidExtensibleArraySignature { .. }));
    }

    #[test]
    fn super_block_layout_matches_libhdf5() {
        // The creation parameters libhdf5 uses for chunk indexes.
        let header = EaHeader {
            client_id: 0,
            element_size: 8,
            max_nelmts_bits: 32,
            idx_blk_elmts: 4,
            data_blk_min_elmts: 16,
            sec_blk_min_data_ptrs: 4,
            max_dblk_page_nelmts_bits: 10,
            _nelmts: 0,
            index_block_address: 0,
        };
        let layout = compute_super_block_layout(&header).unwrap();
        // 1 + 32 - log2(16) super blocks; super block u holds 2^(u/2) data
        // blocks of 2^((u+1)/2) * 16 elements.
        assert_eq!(layout.len(), 29);
        assert_eq!(
            &layout[..6],
            &[(16, 1), (32, 1), (32, 2), (64, 2), (64, 4), (128, 4)]
        );
        assert_eq!(layout[13], (2048, 64));
        // The index block points to the 6 data blocks of super blocks 0-3
        // (elements 4-243) and to the secondary blocks of the other 25.
        assert_eq!(index_block_direct_blocks(&header, &layout), (4, 6));
        let direct: u64 = layout[..4].iter().map(|(elmts, num)| elmts * num).sum();
        assert_eq!(direct, 240);
        assert_eq!(index_block_len(&header, &layout, 8).unwrap(), 298);
        // Data blocks of super block 13 are the first ones split into pages.
        assert_eq!(data_block_pages(1024, layout[12].0).unwrap(), 0);
        assert_eq!(data_block_pages(1024, layout[13].0).unwrap(), 2);
    }

    #[test]
    fn invalid_creation_parameters_are_errors() {
        let mut header = EaHeader {
            client_id: 0,
            element_size: 8,
            max_nelmts_bits: 32,
            idx_blk_elmts: 4,
            data_blk_min_elmts: 16,
            sec_blk_min_data_ptrs: 4,
            max_dblk_page_nelmts_bits: 10,
            _nelmts: 0,
            index_block_address: 0,
        };
        header.data_blk_min_elmts = 12;
        assert!(compute_super_block_layout(&header).is_err());
        header.data_blk_min_elmts = 16;
        header.sec_blk_min_data_ptrs = 0;
        assert!(compute_super_block_layout(&header).is_err());
        header.sec_blk_min_data_ptrs = 4;
        header.max_nelmts_bits = 3;
        assert!(compute_super_block_layout(&header).is_err());
        header.max_nelmts_bits = 65;
        assert!(compute_super_block_layout(&header).is_err());
    }

    #[test]
    fn full_scan_reads_direct_secondary_and_paged_data_blocks() {
        let storage = storage_fixture();

        let entries = collect_extensible_array_chunk_entries_storage(
            &storage,
            TEST_HEADER_ADDR,
            TEST_OFFSET_SIZE,
            TEST_LENGTH_SIZE,
            &[31],
            None,
            &[1],
            None,
        )
        .unwrap();
        assert_eq!(entry_addresses(&entries), expected_addresses());

        let entries = collect_extensible_array_chunk_entries(
            &storage.data,
            TEST_HEADER_ADDR,
            TEST_OFFSET_SIZE,
            TEST_LENGTH_SIZE,
            &[31],
            None,
            &[1],
            None,
        )
        .unwrap();
        assert_eq!(entry_addresses(&entries), expected_addresses());
    }

    #[test]
    fn bounded_scan_finds_each_element() {
        let storage = storage_fixture();

        for idx in 0..31u64 {
            let bounds = (&[idx][..], &[idx][..]);
            let expected: Vec<(u64, u64)> = if TEST_SET_ELEMENTS.contains(&idx) {
                vec![(idx, 0x1000 + idx)]
            } else {
                Vec::new()
            };

            let entries = collect_extensible_array_chunk_entries_storage(
                &storage,
                TEST_HEADER_ADDR,
                TEST_OFFSET_SIZE,
                TEST_LENGTH_SIZE,
                &[31],
                None,
                &[1],
                Some(bounds),
            )
            .unwrap();
            assert_eq!(entry_addresses(&entries), expected, "element {idx}");

            let entries = collect_extensible_array_chunk_entries(
                &storage.data,
                TEST_HEADER_ADDR,
                TEST_OFFSET_SIZE,
                TEST_LENGTH_SIZE,
                &[31],
                None,
                &[1],
                Some(bounds),
            )
            .unwrap();
            assert_eq!(entry_addresses(&entries), expected, "element {idx}");
        }
    }

    #[test]
    fn full_scan_stops_at_the_last_chunk() {
        let storage = storage_fixture();

        // Elements 9 and up are past the dataset's chunks: the paged super
        // block is never read.
        let entries = collect_extensible_array_chunk_entries_storage(
            &storage,
            TEST_HEADER_ADDR,
            TEST_OFFSET_SIZE,
            TEST_LENGTH_SIZE,
            &[9],
            None,
            &[1],
            None,
        )
        .unwrap();
        assert_eq!(
            entry_addresses(&entries),
            vec![
                (0, 0x1000),
                (1, 0x1001),
                (2, 0x1002),
                (7, 0x1007),
                (8, 0x1008)
            ]
        );
        assert!(storage
            .ranges()
            .iter()
            .all(|(offset, _)| *offset < TEST_PAGED_SECONDARY_ADDR));
    }

    #[test]
    fn storage_scans_read_exact_block_lengths() {
        let storage = storage_fixture();

        collect_extensible_array_chunk_entries_storage(
            &storage,
            TEST_HEADER_ADDR,
            TEST_OFFSET_SIZE,
            TEST_LENGTH_SIZE,
            &[31],
            None,
            &[1],
            None,
        )
        .unwrap();
        let ranges = storage.ranges();
        assert!(ranges.contains(&(TEST_HEADER_ADDR, 72)));
        // prefix 14 + 1 inline entry + 2 data block and 6 secondary block
        // addresses + checksum
        assert!(ranges.contains(&(TEST_INDEX_ADDR, 90)));
        // prefix 15 + 2 addresses + checksum; with a 2-byte page bitmap
        assert!(ranges.contains(&(TEST_SECONDARY_ADDR, 35)));
        assert!(ranges.contains(&(TEST_PAGED_SECONDARY_ADDR, 37)));
        // Only the written pages of paged data blocks: after the 15-byte
        // prefix and its checksum, pages of 4 entries and a checksum.
        assert!(ranges.contains(&(TEST_PAGED_DATA_BLOCK_ADDRS[0] + 19 + 36, 32)));
        assert!(ranges.contains(&(TEST_PAGED_DATA_BLOCK_ADDRS[1] + 19, 32)));
        assert!(ranges.iter().all(|(_, len)| *len <= 90), "{ranges:?}");

        storage.clear_ranges();
        let entries = collect_extensible_array_chunk_entries_storage(
            &storage,
            TEST_HEADER_ADDR,
            TEST_OFFSET_SIZE,
            TEST_LENGTH_SIZE,
            &[31],
            None,
            &[1],
            Some((&[8], &[8])),
        )
        .unwrap();
        assert_eq!(entry_addresses(&entries), vec![(8, 0x1008)]);
        let ranges = storage.ranges();
        assert!(ranges.contains(&(TEST_INDEX_ADDR, 90)));
        assert!(ranges.contains(&(TEST_SECONDARY_ADDR, 35)));
        assert!(ranges.contains(&(TEST_DATA_BLOCK_ADDR + 15 + 8, 8)));
        assert!(ranges.iter().all(|(_, len)| *len <= 90), "{ranges:?}");
    }
}
