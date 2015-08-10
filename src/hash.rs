use std::u32;

pub fn simple_hash(s: &[u8]) -> u32 {

    let m = u32::MAX as u64;
    s.iter().fold::<u64, _>(0, |acc, x| (acc * 31 + (*x as u64)) & m) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash() {
        assert_eq!(simple_hash("a".as_bytes()), 97);
        assert_eq!(simple_hash("Joe Miller".as_bytes()), 149190249);
    }
}

