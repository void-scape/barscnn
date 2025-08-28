use super::Image;
use super::pixel::Pixel;

pub struct BmpReader<'a> {
    pub width: u32,
    pub height: u32,
    pub bpp: u32,
    pub compression: Compression,
    pub pixels: &'a [u8],
}

pub fn from_bytes(bytes: &[u8]) -> Result<BmpReader<'_>, &'static str> {
    bmp_from_bytes(bytes)
}

pub struct BmpWriter<'a, Pixel> {
    image: &'a Image<Pixel>,
}

impl<'a, Pixel> BmpWriter<'a, Pixel> {
    pub fn new(image: &'a Image<Pixel>) -> Self {
        Self { image }
    }
}

pub fn to_bytes<T>(writer: BmpWriter<'_, T>) -> Vec<u8>
where
    T: Pixel,
{
    bmp_to_bytes(writer)
}

pub enum Compression {
    Rgb,
    BitFields { r: u32, g: u32, b: u32 },
}

fn bmp_from_bytes(bytes: &[u8]) -> Result<BmpReader<'_>, &'static str> {
    let mut bmp = bytes;
    let input = &mut bmp;

    if input.len() < 14 {
        return Err("BMP header is too small");
    }

    fn read_slice<const LEN: usize>(slice: &mut &[u8]) -> [u8; LEN] {
        let mut data = [0; LEN];
        data.copy_from_slice(&slice[0..LEN]);
        *slice = &slice[LEN..];
        data
    }

    fn u16(input: &mut &[u8]) -> Result<u16, &'static str> {
        if input.len() < 2 {
            return Err("Unexpected end of file");
        }
        Ok(u16::from_le_bytes(read_slice::<2>(input)))
    }

    fn u32(input: &mut &[u8]) -> Result<u32, &'static str> {
        if input.len() < 4 {
            return Err("Unexpected end of file");
        }
        Ok(u32::from_le_bytes(read_slice::<4>(input)))
    }

    fn i32(input: &mut &[u8]) -> Result<i32, &'static str> {
        if input.len() < 4 {
            return Err("Unexpected end of file");
        }
        Ok(i32::from_le_bytes(read_slice::<4>(input)))
    }

    let header = read_slice::<2>(input);
    if header != [0x42, 0x4D] {
        return Err("Invalid BMP header");
    }
    let _size = u32(input)?;
    let _reserved = read_slice::<4>(input);
    let data_index = u32(input)?;

    let header_size = u32(input)?;
    if header_size != 124 {
        return Err(String::leak(format!(
            "Unrecognized BMP header, size is {header_size}"
        )));
    }
    let width = i32(input)?;
    let height = i32(input)?;

    let color_panes = u16(input)?;
    if color_panes != 1 {
        return Err("Invalid BMP header");
    }
    let bpp = u16(input)? as usize;

    let compression = match u32(input)? {
        0 => {
            if bpp != 24 {
                return Err(String::leak(format!("Unsupported bits per pixel: {bpp}")));
            }

            let current_index = 0x22;
            let skip = data_index - current_index;
            *input = &input[skip as usize..];

            Compression::Rgb
        }
        3 => {
            if bpp != 32 {
                return Err(String::leak(format!("Unsupported bits per pixel: {bpp}")));
            }

            *input = &input[20..];
            let r = u32(input)?;
            let g = u32(input)?;
            let b = u32(input)?;

            let current_index = 0x42;
            let skip = data_index - current_index;
            *input = &input[skip as usize..];

            Compression::BitFields { r, g, b }
        }
        c => {
            return Err(String::leak(format!("Unsupported compression: {c}")));
        }
    };

    Ok(BmpReader {
        width: width as u32,
        height: height as u32,
        bpp: bpp as u32,
        compression,
        pixels: *input,
    })
}

pub fn bmp_to_bytes<T>(writer: BmpWriter<'_, T>) -> Vec<u8>
where
    T: Pixel,
{
    let width = writer.image.width as usize;
    let height = writer.image.height as usize;

    let bytes_per_row = width * 3;
    let padding = (4 - (bytes_per_row % 4)) % 4;
    let padded_row_size = bytes_per_row + padding;
    let image_size = padded_row_size * height;

    let mut bytes = Vec::new();

    bytes.extend([0x42, 0x4D]);
    bytes.extend(u32::to_le_bytes(54 + image_size as u32));
    bytes.extend([0; 4]);
    bytes.extend(u32::to_le_bytes(54));

    // DIB HEADER (BITMAPINFOHEADER)
    bytes.extend(u32::to_le_bytes(40));
    bytes.extend(i32::to_le_bytes(width as i32));
    bytes.extend(i32::to_le_bytes(height as i32));
    bytes.extend(u16::to_le_bytes(1));
    // bpp
    bytes.extend(u16::to_le_bytes(24));
    // cmp (BI_RGB)
    bytes.extend(u32::to_le_bytes(0));
    bytes.extend(u32::to_le_bytes(image_size as u32));
    bytes.extend(u32::to_le_bytes(0));
    bytes.extend(u32::to_le_bytes(0));
    bytes.extend(u32::to_le_bytes(0));
    bytes.extend(u32::to_le_bytes(0));
    assert_eq!(bytes.len(), 54);

    for y in (0..height).rev() {
        for x in 0..width {
            let pixel_index = y * width + x;
            let rgb = writer.image.pixels[pixel_index].to_rgb();
            bytes.push(rgb[2]);
            bytes.push(rgb[1]);
            bytes.push(rgb[0]);
        }
        for _ in 0..padding {
            bytes.push(0);
        }
    }

    bytes
}
