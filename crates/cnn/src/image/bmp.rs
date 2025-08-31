use super::Image;

pub fn from_bytes(bytes: &[u8]) -> Result<Image, &'static str> {
    image_from_bytes(bytes)
}

pub fn to_bytes(image: &Image) -> Vec<u8> {
    image_to_bytes(image)
}

fn image_from_bytes(bytes: &[u8]) -> Result<Image, &'static str> {
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
    if header_size != 124 && header_size != 40 {
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

    pub enum Compression {
        Grayscale,
        Rgb,
        BitFields { r: u32, g: u32, b: u32 },
    }

    let compression = match u32(input)? {
        0 => {
            let current_index = 0x22;
            let skip = data_index - current_index;
            *input = &input[skip as usize..];

            match bpp {
                8 => Compression::Grayscale,
                24 => Compression::Rgb,
                _ => return Err(String::leak(format!("Unsupported bits per pixel: {bpp}"))),
            }
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

    let width = width as usize;
    let height = height as usize;

    assert!(bpp % 8 == 0);
    let mut pixels = Vec::with_capacity(width * height * bpp / 8);

    match compression {
        Compression::Rgb => {
            assert_eq!(bpp, 24);

            let bpr = width as usize * 3;
            let padding = (4 - (bpr % 4)) % 4;

            for h in (0..height).rev() {
                *input = &bytes[h * 3 * width + padding..];

                for _ in 0..width {
                    let b = input[0] as f32 / u8::MAX as f32;
                    let g = input[1] as f32 / u8::MAX as f32;
                    let r = input[2] as f32 / u8::MAX as f32;
                    *input = &input[3..];
                    pixels.extend([r, g, b]);
                }
            }
        }
        Compression::BitFields { r, g, b } => {
            let row_size = ((bpp * width) as f32 / 32.0).ceil() as usize;
            assert_eq!(width as usize, row_size);
            assert_eq!(bpp, 32);

            fn extract_channel(pixel: u32, mask: u32) -> u8 {
                if mask == 0 {
                    return 0;
                }
                let shift = mask.trailing_zeros();
                let value = (pixel & mask) >> shift;
                let max_value = mask >> shift;
                ((value * 255) / max_value) as u8
            }

            for h in (0..height as usize).rev() {
                *input = &bytes[h * 4 * width as usize..];

                for _ in 0..width {
                    let pixel = u32(input)?;
                    let r = extract_channel(pixel, r) as f32 / u8::MAX as f32;
                    let g = extract_channel(pixel, g) as f32 / u8::MAX as f32;
                    let b = extract_channel(pixel, b) as f32 / u8::MAX as f32;
                    pixels.extend([r, g, b]);
                }
            }
        }
        Compression::Grayscale => {
            assert_eq!(bpp, 8);
            let bpr = width as usize;
            let padding = (4 - (bpr % 4)) % 4;

            for h in (0..height as usize).rev() {
                *input = &bytes[h * width as usize + padding..];

                for _ in 0..width {
                    let l = input[0];
                    *input = &input[1..];
                    pixels.push(l as f32 / u8::MAX as f32);
                }
            }
        }
    }

    Ok(Image {
        width,
        height,
        channels: bpp / 8,
        pixels,
    })
}

fn image_to_bytes(image: &Image) -> Vec<u8> {
    let width = image.width as usize;
    let height = image.height as usize;

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
            match image.channels {
                1 => {
                    let pixel = (image.pixels[y * width + x].clamp(0.0, u8::MAX as f32)
                        / u8::MAX as f32) as u8;
                    bytes.extend([pixel; 3]);
                }
                3 => {
                    let r = (image.pixels[y * width + x].clamp(0.0, u8::MAX as f32)
                        / u8::MAX as f32) as u8;
                    let g = (image.pixels[y * width + x].clamp(0.0, u8::MAX as f32)
                        / u8::MAX as f32) as u8;
                    let b = (image.pixels[y * width + x].clamp(0.0, u8::MAX as f32)
                        / u8::MAX as f32) as u8;
                    bytes.extend([b, g, r]);
                }
                c => panic!("invalid BMP channels: {c}"),
            }
        }
        for _ in 0..padding {
            bytes.push(0);
        }
    }

    bytes
}
