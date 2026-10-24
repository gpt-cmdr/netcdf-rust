//! HDF5 Attribute message (type 0x000C).
//!
//! Attributes are small named data items attached to any HDF5 object.
//! Each attribute has a name, datatype, dataspace, and raw data.

use crate::error::{Error, Result};
use crate::io::Cursor;

use super::dataspace::{self, DataspaceMessage};
use super::datatype::{self, Datatype};

/// Parsed attribute message.
#[derive(Debug, Clone)]
pub struct AttributeMessage {
    /// Attribute name.
    pub name: String,
    /// Datatype of the attribute value.
    pub datatype: Datatype,
    /// Dataspace describing the shape of the attribute data.
    pub dataspace: DataspaceMessage,
    /// Raw attribute data (un-decoded bytes).
    pub raw_data: Vec<u8>,
}

/// Parse an attribute message.
///
/// Attribute messages embed inline datatype and dataspace descriptions
/// followed by the raw data bytes.
pub fn parse(
    cursor: &mut Cursor<'_>,
    offset_size: u8,
    length_size: u8,
    msg_size: usize,
) -> Result<AttributeMessage> {
    // The embedded datatype, dataspace and data must lie within the message:
    // parse from the message's own bytes, never from the messages after it.
    let mut message = Cursor::new(cursor.read_bytes(msg_size)?);
    let version = message.read_u8()?;
    match version {
        1 => parse_v1(&mut message, offset_size, length_size),
        2 => parse_v2(&mut message, offset_size, length_size),
        3 => parse_v3(&mut message, offset_size, length_size),
        v => Err(Error::UnsupportedAttributeVersion(v)),
    }
}

fn parse_v1(cursor: &mut Cursor<'_>, offset_size: u8, length_size: u8) -> Result<AttributeMessage> {
    let _reserved = cursor.read_u8()?;
    let name_size = cursor.read_u16_le()? as usize;
    let datatype_size = cursor.read_u16_le()? as usize;
    let dataspace_size = cursor.read_u16_le()? as usize;

    // Name — padded to 8-byte boundary
    let name = cursor.read_fixed_string(name_size)?;
    let name_padded = (name_size + 7) & !7;
    if name_padded > name_size {
        cursor.skip(name_padded - name_size)?;
    }

    // Datatype — padded to 8
    let dt_start = cursor.position();
    let dt_msg = datatype::parse(cursor, datatype_size)?;
    if let Some(raw) = super::raw_bytes(cursor, dt_start, datatype_size) {
        super::record_attribute_datatype(&name, raw);
    }
    let dt_consumed = datatype_size; // parse() already handles its own size
    let dt_padded = (dt_consumed + 7) & !7;
    if dt_padded > dt_consumed {
        cursor.skip(dt_padded - dt_consumed)?;
    }

    // Dataspace — padded to 8
    let ds_msg = dataspace::parse(cursor, offset_size, length_size, dataspace_size)?;
    let ds_consumed = dataspace_size;
    let ds_padded = (ds_consumed + 7) & !7;
    if ds_padded > ds_consumed {
        cursor.skip(ds_padded - ds_consumed)?;
    }

    // Raw data — remaining bytes are the attribute data
    let data_size = attribute_raw_data_size(&ds_msg, dt_msg.size)?;
    let raw_data = if data_size > 0 {
        cursor.read_bytes(data_size)?.to_vec()
    } else {
        vec![]
    };

    Ok(AttributeMessage {
        name,
        datatype: dt_msg.datatype,
        dataspace: ds_msg,
        raw_data,
    })
}

fn parse_v2(cursor: &mut Cursor<'_>, offset_size: u8, length_size: u8) -> Result<AttributeMessage> {
    let flags = cursor.read_u8()?;
    check_flags(flags, 2)?;
    let name_size = cursor.read_u16_le()? as usize;
    let datatype_size = cursor.read_u16_le()? as usize;
    let dataspace_size = cursor.read_u16_le()? as usize;

    // Name — NOT padded in v2
    let name = cursor.read_fixed_string(name_size)?;

    // Datatype
    let dt_start = cursor.position();
    let dt_msg = datatype::parse(cursor, datatype_size)?;
    if let Some(raw) = super::raw_bytes(cursor, dt_start, datatype_size) {
        super::record_attribute_datatype(&name, raw);
    }

    // Dataspace
    let ds_msg = dataspace::parse(cursor, offset_size, length_size, dataspace_size)?;

    // Raw data
    let data_size = attribute_raw_data_size(&ds_msg, dt_msg.size)?;
    let raw_data = if data_size > 0 {
        cursor.read_bytes(data_size)?.to_vec()
    } else {
        vec![]
    };

    Ok(AttributeMessage {
        name,
        datatype: dt_msg.datatype,
        dataspace: ds_msg,
        raw_data,
    })
}

fn parse_v3(cursor: &mut Cursor<'_>, offset_size: u8, length_size: u8) -> Result<AttributeMessage> {
    let flags = cursor.read_u8()?;
    let name_size = cursor.read_u16_le()? as usize;
    let datatype_size = cursor.read_u16_le()? as usize;
    let dataspace_size = cursor.read_u16_le()? as usize;
    let _name_encoding = cursor.read_u8()?;
    check_flags(flags, 3)?;

    // Name — NOT padded in v3
    let name = cursor.read_fixed_string(name_size)?;

    // Datatype
    let dt_start = cursor.position();
    let dt_msg = datatype::parse(cursor, datatype_size)?;
    if let Some(raw) = super::raw_bytes(cursor, dt_start, datatype_size) {
        super::record_attribute_datatype(&name, raw);
    }

    // Dataspace
    let ds_msg = dataspace::parse(cursor, offset_size, length_size, dataspace_size)?;

    // Raw data
    let data_size = attribute_raw_data_size(&ds_msg, dt_msg.size)?;
    let raw_data = if data_size > 0 {
        cursor.read_bytes(data_size)?.to_vec()
    } else {
        vec![]
    };

    Ok(AttributeMessage {
        name,
        datatype: dt_msg.datatype,
        dataspace: ds_msg,
        raw_data,
    })
}

/// Version 2 and 3 attribute flags: bit 0 marks a shared datatype, bit 1 a
/// shared dataspace; the other bits are reserved.
fn check_flags(flags: u8, version: u8) -> Result<()> {
    if flags & !0x03 != 0 {
        return Err(Error::InvalidData(format!(
            "reserved flags {flags:#04x} in attribute v{version}"
        )));
    }
    if flags != 0 {
        return Err(Error::UnsupportedEncoding(format!(
            "shared datatype/dataspace in attribute v{version}"
        )));
    }
    Ok(())
}

fn attribute_raw_data_size(ds_msg: &dataspace::DataspaceMessage, dtype_size: u32) -> Result<usize> {
    let elements = usize::try_from(ds_msg.num_elements()?).map_err(|_| {
        Error::InvalidData("attribute element count exceeds platform usize capacity".to_string())
    })?;
    elements.checked_mul(dtype_size as usize).ok_or_else(|| {
        Error::InvalidData("attribute raw data size exceeds platform usize capacity".to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ByteOrder;
    use crate::messages::dataspace::DataspaceType;

    /// Build a minimal u32-LE datatype description (8 bytes total: 4 class/ver/flags + 4 size).
    fn u32_le_datatype() -> Vec<u8> {
        let mut buf = Vec::new();
        // class=0, version=1, flags=0 (LE unsigned)
        let class_word: u32 = 0x01 << 4;
        buf.extend_from_slice(&class_word.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes()); // size=4
                                                    // properties: offset=0, precision=32
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&32u16.to_le_bytes());
        buf
    }

    /// Build a scalar dataspace (v2, rank=0, type=scalar).
    fn scalar_dataspace() -> Vec<u8> {
        vec![0x02, 0x00, 0x00, 0x00]
    }

    /// A version 2 or 3 scalar u32 attribute "abc" = 99.
    fn attr(version: u8, flags: u8) -> Vec<u8> {
        let dt = u32_le_datatype();
        let ds = scalar_dataspace();
        let mut data = vec![version, flags];
        data.extend_from_slice(&4u16.to_le_bytes());
        data.extend_from_slice(&(dt.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds.len() as u16).to_le_bytes());
        if version == 3 {
            data.push(0x00); // ASCII name
        }
        data.extend_from_slice(b"abc\0");
        data.extend_from_slice(&dt);
        data.extend_from_slice(&ds);
        data.extend_from_slice(&99u32.to_le_bytes());
        data
    }

    #[test]
    fn reserved_and_shared_attribute_flags_are_refused() {
        for version in [2, 3] {
            let data = attr(version, 0);
            assert_eq!(
                parse(&mut Cursor::new(&data), 8, 8, data.len())
                    .unwrap()
                    .raw_data,
                99u32.to_le_bytes()
            );
            for flags in [0x04, 0x80] {
                let data = attr(version, flags);
                let err = parse(&mut Cursor::new(&data), 8, 8, data.len()).unwrap_err();
                assert!(
                    err.to_string().contains("reserved flags"),
                    "v{version}: {err}"
                );
            }
            // Shared datatype or dataspace: the inline fields are not there.
            for flags in [0x01, 0x02, 0x03] {
                let data = attr(version, flags);
                let err = parse(&mut Cursor::new(&data), 8, 8, data.len()).unwrap_err();
                assert!(err.to_string().contains("shared"), "v{version}: {err}");
            }
        }
    }

    #[test]
    fn attribute_is_parsed_within_its_message() {
        let data = attr(3, 0);

        // The value runs two bytes past the declared message size: an error,
        // not two bytes of the next message.
        let mut overrun = data.clone();
        overrun.extend_from_slice(&[0xaa; 8]);
        let mut cursor = Cursor::new(&overrun);
        assert!(parse(&mut cursor, 8, 8, data.len() - 2).is_err());

        // Trailing padding inside the message is skipped.
        let mut padded = data.clone();
        padded.extend_from_slice(&[0; 4]);
        padded.extend_from_slice(&[0xaa; 4]);
        let mut cursor = Cursor::new(&padded);
        let msg = parse(&mut cursor, 8, 8, data.len() + 4).unwrap();
        assert_eq!(msg.raw_data, 99u32.to_le_bytes());
        assert_eq!(cursor.position() as usize, data.len() + 4);
    }

    #[test]
    fn datatype_larger_than_its_message_is_refused() {
        let dt = u32_le_datatype();
        assert!(datatype::parse(&mut Cursor::new(&dt), dt.len()).is_ok());
        let err = datatype::parse(&mut Cursor::new(&dt), dt.len() - 4).unwrap_err();
        assert!(err.to_string().contains("exceeds"), "{err}");
    }

    #[test]
    fn parse_v1_scalar_u32_attr() {
        let dt = u32_le_datatype();
        let ds = scalar_dataspace();

        let mut data = vec![
            0x01, // version 1
            0x00, // reserved
        ];
        // name size = 5 ("temp\0")
        data.extend_from_slice(&5u16.to_le_bytes());
        // datatype size
        data.extend_from_slice(&(dt.len() as u16).to_le_bytes());
        // dataspace size
        data.extend_from_slice(&(ds.len() as u16).to_le_bytes());

        // Name "temp\0" padded to 8 bytes
        data.extend_from_slice(b"temp\0\0\0\0");

        // Datatype (12 bytes), padded to 16
        data.extend_from_slice(&dt);
        data.extend_from_slice(&[0u8; 4]); // padding to 16

        // Dataspace (4 bytes), padded to 8
        data.extend_from_slice(&ds);
        data.extend_from_slice(&[0u8; 4]); // padding to 8

        // Raw data: 1 scalar element * 4 bytes
        data.extend_from_slice(&42u32.to_le_bytes());

        let mut cursor = Cursor::new(&data);
        let msg = parse(&mut cursor, 8, 8, data.len()).unwrap();
        assert_eq!(msg.name, "temp");
        assert_eq!(msg.dataspace.dataspace_type, DataspaceType::Scalar);
        assert_eq!(msg.raw_data, 42u32.to_le_bytes());
        match &msg.datatype {
            Datatype::FixedPoint {
                size: 4,
                signed: false,
                byte_order: ByteOrder::LittleEndian,
            } => {}
            other => panic!("unexpected datatype: {:?}", other),
        }
    }

    fn v3_attr(flags: u8) -> Vec<u8> {
        let dt = u32_le_datatype();
        let ds = scalar_dataspace();
        let mut data = vec![0x03, flags];
        data.extend_from_slice(&4u16.to_le_bytes());
        data.extend_from_slice(&(dt.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds.len() as u16).to_le_bytes());
        data.push(0x00);
        data.extend_from_slice(b"abc\0");
        data.extend_from_slice(&dt);
        data.extend_from_slice(&ds);
        data.extend_from_slice(&99u32.to_le_bytes());
        data
    }

    #[test]
    fn shared_v3_attribute_encodings_are_unsupported_not_corrupt() {
        for flags in [1, 2, 3] {
            let data = v3_attr(flags);
            let err = parse(&mut Cursor::new(&data), 8, 8, data.len()).unwrap_err();
            assert!(matches!(err, Error::UnsupportedEncoding(_)), "{err}");
        }
    }

    #[test]
    fn attribute_datatype_bytes_are_recorded_only_while_capturing() {
        let data = v3_attr(0);
        parse(&mut Cursor::new(&data), 8, 8, data.len()).unwrap();

        let outer = crate::messages::RawCapture::start();
        let inner = crate::messages::RawCapture::start();
        parse(&mut Cursor::new(&data), 8, 8, data.len()).unwrap();
        let captured = inner.finish();
        assert_eq!(
            captured.attributes,
            vec![("abc".to_string(), u32_le_datatype())]
        );
        // The outer capture is active again and saw nothing of the inner one.
        parse(&mut Cursor::new(&data), 8, 8, data.len()).unwrap();
        assert_eq!(outer.finish().attributes.len(), 1);

        // A capture dropped without finishing leaves none active.
        drop(crate::messages::RawCapture::start());
        let after = crate::messages::RawCapture::start();
        assert!(after.finish().attributes.is_empty());
    }

    #[test]
    fn parse_v3_scalar_attr() {
        let dt = u32_le_datatype();
        let ds = scalar_dataspace();

        let mut data = vec![
            0x03, // version 3
            0x00, // flags
        ];
        // name size = 4 ("abc\0")
        data.extend_from_slice(&4u16.to_le_bytes());
        data.extend_from_slice(&(dt.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds.len() as u16).to_le_bytes());
        data.push(0x00); // ASCII name encoding

        // Name (not padded in v3)
        data.extend_from_slice(b"abc\0");

        // Datatype
        data.extend_from_slice(&dt);

        // Dataspace
        data.extend_from_slice(&ds);

        // Raw data
        data.extend_from_slice(&99u32.to_le_bytes());

        let mut cursor = Cursor::new(&data);
        let msg = parse(&mut cursor, 8, 8, data.len()).unwrap();
        assert_eq!(msg.name, "abc");
        assert_eq!(msg.dataspace.dataspace_type, DataspaceType::Scalar);
        assert_eq!(msg.raw_data, 99u32.to_le_bytes());
    }

    #[test]
    fn parse_v3_utf8_name_attr() {
        let dt = u32_le_datatype();
        let ds = scalar_dataspace();

        let mut data = vec![
            0x03, // version 3
            0x00, // flags
        ];
        data.extend_from_slice(&2u16.to_le_bytes()); // "x\0"
        data.extend_from_slice(&(dt.len() as u16).to_le_bytes());
        data.extend_from_slice(&(ds.len() as u16).to_le_bytes());
        data.push(0x01); // UTF-8 name encoding
        data.extend_from_slice(b"x\0");
        data.extend_from_slice(&dt);
        data.extend_from_slice(&ds);
        data.extend_from_slice(&7u32.to_le_bytes());

        let mut cursor = Cursor::new(&data);
        let msg = parse(&mut cursor, 8, 8, data.len()).unwrap();
        assert_eq!(msg.name, "x");
        assert_eq!(msg.dataspace.dataspace_type, DataspaceType::Scalar);
        assert_eq!(msg.raw_data, 7u32.to_le_bytes());
    }
}
