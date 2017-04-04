use std::fmt;

const FILE_MASK: [Bitboard; 8] = [
    Bitboard(0x0101010101010101),
    Bitboard(0x0202020202020202),
    Bitboard(0x0404040404040404),
    Bitboard(0x0808080808080808),
    Bitboard(0x1010101010101010),
    Bitboard(0x2020202020202020),
    Bitboard(0x4040404040404040),
    Bitboard(0x8080808080808080),
];

const RANK_MASK: [Bitboard; 8] = [
    Bitboard(0x00000000000000FF),
    Bitboard(0x000000000000FF00),
    Bitboard(0x0000000000FF0000),
    Bitboard(0x00000000FF000000),
    Bitboard(0x000000FF00000000),
    Bitboard(0x0000FF0000000000),
    Bitboard(0x00FF000000000000),
    Bitboard(0xFF00000000000000),
];

// Diagonal enumerations based off
// https://chessprogramming.wikispaces.com/diagonals
// https://chessprogramming.wikispaces.com/Anti-Diagonals

const DIAGONAL_MASK: [Bitboard; 16] = [
    Bitboard(0x0000000000000080),
    Bitboard(0x0000000000008040),
    Bitboard(0x0000000000804020),
    Bitboard(0x0000000080402010),
    Bitboard(0x0000008040201008),
    Bitboard(0x0000804020100804),
    Bitboard(0x0080402010080402),
    Bitboard(0x8040201008040201),
    Bitboard(0x4020100804020100),
    Bitboard(0x2010080402010000),
    Bitboard(0x1008040201000000),
    Bitboard(0x0804020100000000),
    Bitboard(0x0402010000000000),
    Bitboard(0x0201000000000000),
    Bitboard(0x0100000000000000),
    Bitboard(0x0000000000000000),
];

const ANTI_DIAGONAL_MASK: [Bitboard; 16] = [
    Bitboard(0x0000000000000001),
    Bitboard(0x0000000000000102),
    Bitboard(0x0000000000010204),
    Bitboard(0x0000000001020408),
    Bitboard(0x0000000102040810),
    Bitboard(0x0000010204081020),
    Bitboard(0x0001020408102040),
    Bitboard(0x0102040810204080),
    Bitboard(0x0204081020408000),
    Bitboard(0x0408102040800000),
    Bitboard(0x0810204080000000),
    Bitboard(0x1020408000000000),
    Bitboard(0x2040800000000000),
    Bitboard(0x4080000000000000),
    Bitboard(0x8000000000000000),
    Bitboard(0x0000000000000000),
];

// We are going to use a bitboard scheme from here:
// https://chessprogramming.wikispaces.com/Square+Mapping+Considerations
// We are using LERF (Little-Endian Rank-File)
#[derive(Clone,Copy,PartialEq,Eq,Hash)]
pub struct Bitboard(u64);

// TODO: use self or &self for functions?
impl Bitboard
{
    pub fn new(r7: u64, r6: u64, r5: u64, r4: u64, r3: u64, r2: u64, r1: u64, r0: u64) -> Bitboard
    {
        Bitboard(
            (((((((r7 as u64) << 8 |
                   r6 as u64) << 8 |
                   r5 as u64) << 8 |
                   r4 as u64) << 8 |
                   r3 as u64) << 8 |
                   r2 as u64) << 8 |
                   r1 as u64) << 8 |
                   r0 as u64)
    }

    pub fn from_u64(board: u64) -> Bitboard
    {
        Bitboard(board)
    }

    pub fn is_empty(&self) -> bool
    {
        self.0 == 0
    }

    // Should make a BitboardPiece and use .contains() instead
    pub fn is_set(&self, x: u32, y: u32) -> bool
    {
        // self.0 & ((1 << (y * 8)) << x) != 0
        self.0 & (1 << (y * 8 + x)) != 0
    }

    pub fn contains(&self, piece: BitboardPiece) -> bool
    {
        self.0 & piece.0 != 0
    }

    pub fn remove(&mut self, piece: BitboardPiece)
    {
        self.0 &= !piece.0;
    }

    pub fn add(&mut self, piece: BitboardPiece)
    {
        self.0 |= piece.0;
    }

    pub fn union(self, other: Bitboard) -> Bitboard
    {
        Bitboard(self.0 | other.0)
    }

    pub fn intersect(self, other: Bitboard) -> Bitboard
    {
        Bitboard(self.0 & other.0)
    }

    pub fn complement(self) -> Bitboard
    {
        Bitboard(!self.0)
    }

    pub fn flip_vertical(self) -> Bitboard
    {
        Bitboard(self.0.swap_bytes())
    }

    // Algorithm from
    // https://chessprogramming.wikispaces.com/Flipping+Mirroring+and+Rotating
    pub fn mirror_horizontal(self) -> Bitboard
    {
        const K1: u64 = 0x5555555555555555;
        const K2: u64 = 0x3333333333333333;
        const K4: u64 = 0x0f0f0f0f0f0f0f0f;
        let mut x = self.0;
        x = ((x >> 1) & K1) | ((x & K1) << 1);
        x = ((x >> 2) & K2) | ((x & K2) << 2);
        x = ((x >> 4) & K4) | ((x & K4) << 4);
        Bitboard(x)
    }

    pub fn positive_horizontal_ray(self, piece: BitboardPiece) -> Bitboard
    {
        // TODO: mask row
        Bitboard(self.0 ^ (self.0 - 2 * piece.0)).intersect(RANK_MASK[piece.rank()])
    }

    pub fn negative_horizontal_ray(self, piece: BitboardPiece) -> Bitboard
    {
        // TODO: mask row
        let mirror_board = self.mirror_horizontal();
        let mirror_piece = piece.mirror_horizontal();
        Bitboard(mirror_board.0 ^ (mirror_board.0.wrapping_sub(2 * mirror_piece.0))).mirror_horizontal()
                                                                        .intersect(RANK_MASK[piece.rank()])
    }

    pub fn horizontal_ray(self, piece: BitboardPiece) -> Bitboard
    {
        let o_prime = self.mirror_horizontal();
        let r_prime = piece.mirror_horizontal();
        let temp = Bitboard(o_prime.0.wrapping_sub(2 * r_prime.0)).mirror_horizontal();
        Bitboard((self.0.wrapping_sub(2 * piece.0)) ^ temp.0).intersect(RANK_MASK[piece.rank()])
    }

    pub fn positive_vertical_ray(self, piece: BitboardPiece) -> Bitboard
    {
        // TODO: mask row
        let blockers = self.intersect(FILE_MASK[piece.file()]);
        Bitboard(blockers.0 ^ (blockers.0.wrapping_sub(2 * piece.0))).intersect(FILE_MASK[piece.file()])
    }

    pub fn negative_vertical_ray(self, piece: BitboardPiece) -> Bitboard
    {
        // TODO: mask row
        let mirror_board = self.intersect(FILE_MASK[piece.file()]).flip_vertical();
        let mirror_piece = piece.flip_vertical();
        Bitboard(mirror_board.0 ^ (mirror_board.0.wrapping_sub(2 * mirror_piece.0))).flip_vertical()
                                                                        .intersect(FILE_MASK[piece.file()])
    }

    pub fn vertical_ray(self, piece: BitboardPiece) -> Bitboard
    {
        let o = self.intersect(FILE_MASK[piece.file()]);
        let o_prime = self.intersect(FILE_MASK[piece.file()]).flip_vertical();
        let r_prime = piece.flip_vertical();
        let temp = Bitboard(o_prime.0.wrapping_sub(2 * r_prime.0)).flip_vertical();
        Bitboard((o.0.wrapping_sub(2 * piece.0)) ^ temp.0).intersect(FILE_MASK[piece.file()])
    }

    pub fn positive_diagonal_ray(self, piece: BitboardPiece) -> Bitboard
    {
        // TODO: mask row
        let blockers = self.intersect(DIAGONAL_MASK[piece.diagonal()]);
        Bitboard(blockers.0 ^ (blockers.0.wrapping_sub(2 * piece.0))).intersect(DIAGONAL_MASK[piece.diagonal()])
    }

    pub fn negative_diagonal_ray(self, piece: BitboardPiece) -> Bitboard
    {
        // TODO: mask row
        let mirror_board = self.intersect(DIAGONAL_MASK[piece.diagonal()]).flip_vertical();
        let mirror_piece = piece.flip_vertical();
        Bitboard(mirror_board.0 ^ (mirror_board.0.wrapping_sub(2 * mirror_piece.0))).flip_vertical()
                                                                        .intersect(DIAGONAL_MASK[piece.diagonal()])
    }

    pub fn diagonal_ray(self, piece: BitboardPiece) -> Bitboard
    {
        let o = self.intersect(DIAGONAL_MASK[piece.diagonal()]);
        let o_prime = self.intersect(DIAGONAL_MASK[piece.diagonal()]).flip_vertical();
        let r_prime = piece.flip_vertical();
        let temp = Bitboard(o_prime.0.wrapping_sub(2 * r_prime.0)).flip_vertical();
        Bitboard((o.0.wrapping_sub(2 * piece.0)) ^ temp.0).intersect(DIAGONAL_MASK[piece.diagonal()])
    }

    pub fn positive_anti_diagonal_ray(self, piece: BitboardPiece) -> Bitboard
    {
        // TODO: mask row
        let blockers = self.intersect(ANTI_DIAGONAL_MASK[piece.anti_diagonal()]);
        Bitboard(blockers.0 ^ (blockers.0.wrapping_sub(2 * piece.0))).intersect(ANTI_DIAGONAL_MASK[piece.anti_diagonal()])
    }

    pub fn negative_anti_diagonal_ray(self, piece: BitboardPiece) -> Bitboard
    {
        // TODO: mask row
        let mirror_board = self.intersect(ANTI_DIAGONAL_MASK[piece.anti_diagonal()]).flip_vertical();
        let mirror_piece = piece.flip_vertical();
        Bitboard(mirror_board.0 ^ (mirror_board.0.wrapping_sub(2 * mirror_piece.0))).flip_vertical()
                                                                        .intersect(ANTI_DIAGONAL_MASK[piece.anti_diagonal()])
    }

    pub fn anti_diagonal_ray(self, piece: BitboardPiece) -> Bitboard
    {
        let o = self.intersect(ANTI_DIAGONAL_MASK[piece.anti_diagonal()]);
        let o_prime = self.intersect(ANTI_DIAGONAL_MASK[piece.anti_diagonal()]).flip_vertical();
        let r_prime = piece.flip_vertical();
        let temp = Bitboard(o_prime.0.wrapping_sub(2 * r_prime.0)).flip_vertical();
        Bitboard((o.0.wrapping_sub(2 * piece.0)) ^ temp.0).intersect(ANTI_DIAGONAL_MASK[piece.anti_diagonal()])
    }

    pub fn shift(self, x: i32, y: i32) -> Bitboard
    {
        let shift = 8 * y + x;
        if shift < 0
        {
            Bitboard(self.0 >> -shift)
        }
        else
        {
            Bitboard(self.0 << shift)
        }
    }

    pub fn to_piece(self) -> BitboardPiece
    {
        // assert self.0 is a power of 2 (only one bit set)
        assert!(self.0 != 0 && (self.0 & (self.0 - 1)) == 0);
        BitboardPiece(self.0)
    }

    pub fn pieces(self) -> BitboardPieces
    {
        BitboardPieces(self.0)
    }

    pub fn num_pieces(&self) -> u32
    {
        self.0.count_ones()
    }
}

impl fmt::Debug for Bitboard
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result
    {
        try!(write!(fmt, "\n"));
        for rank in 0..8
        {
            let mut mask = 1 << (8 * (7 - rank));
            for _ in 0..8
            {
                if self.0 & mask != 0
                {
                    try!(write!(fmt, "*"));
                }
                else
                {
                    try!(write!(fmt, "."));
                }
                mask <<= 1;
            }
            try!(write!(fmt, "\n"));
        }
        Ok(())
    }
}

#[derive(Clone,Copy,Debug)]
pub struct BitboardPieces(u64);

impl Iterator for BitboardPieces
{
    type Item = BitboardPiece;

    fn next(&mut self) -> Option<Self::Item>
    {
        if self.0 != 0
        {
            // Get lsb
            let lsb = ((self.0 as i64) & (self.0 as i64).wrapping_neg()) as u64;

            // Remove lsb from Bitboard
            self.0 &= self.0 - 1;

            Some(BitboardPiece(lsb))
        }
        else
        {
            None
        }
    }
}

// Guarenteed to have only one bit set
#[derive(Clone,Copy,PartialEq,Eq)]
pub struct BitboardPiece(u64);

impl BitboardPiece
{
    pub fn from_square(square: usize) -> BitboardPiece
    {
        BitboardPiece(1 << square)
    }

    pub fn from_file_rank(file: usize, rank: usize) -> BitboardPiece
    {
        BitboardPiece(1 << (rank * 8) << file)
    }

    pub fn file(&self) -> usize
    {
        (self.square() & 7) as usize
    }

    pub fn rank(&self) -> usize
    {
        (self.square() >> 3) as usize
    }

    // TODO: Not sure if this should be public
    // because differing ways to enumerate.
    fn diagonal(&self) -> usize
    {
        let square = self.square();
        ((7 + (square >> 3)) - (square & 7))
    }

    // TODO: Not sure if this should be public
    // because differing ways to enumerate.
    fn anti_diagonal(&self) -> usize
    {
        let square = self.square();
        ((square >> 3) + (square & 7))
    }

    pub fn square(&self) -> usize
    {
        self.0.trailing_zeros() as usize
    }

    pub fn as_bitboard(self) -> Bitboard
    {
        Bitboard(self.0)
    }

    pub fn shift(self, x: i32, y: i32) -> BitboardPiece
    {
        let shift = 8 * y + x;
        if shift < 0
        {
            BitboardPiece(self.0 >> -shift)
        }
        else
        {
            BitboardPiece(self.0 << shift)
        }
    }

    pub fn flip_vertical(self) -> BitboardPiece
    {
        BitboardPiece(self.0.swap_bytes())
    }

    // Algorithm from
    // https://chessprogramming.wikispaces.com/Flipping+Mirroring+and+Rotating
    pub fn mirror_horizontal(self) -> BitboardPiece
    {
        const K1: u64 = 0x5555555555555555;
        const K2: u64 = 0x3333333333333333;
        const K4: u64 = 0x0f0f0f0f0f0f0f0f;
        let mut x = self.0;
        x = ((x >> 1) & K1) | ((x & K1) << 1);
        x = ((x >> 2) & K2) | ((x & K2) << 2);
        x = ((x >> 4) & K4) | ((x & K4) << 4);
        BitboardPiece(x)
    }
}

impl fmt::Debug for BitboardPiece
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result
    {
        let square = self.square();
        let x = square & 7;
        let y = square >> 3;
        fmt.debug_struct("BitboardPiece")
            .field("raw", &self.0)
            .field("x", &x)
            .field("y", &y)
            .finish()
    }
}

#[cfg(test)]
mod tests
{
    use super::{Bitboard, BitboardPiece};

    #[test]
    fn test_mirror()
    {
        let board = Bitboard::new(0b00010100,
                                  0b11010100,
                                  0b01001100,
                                  0b10100110,
                                  0b10100101,
                                  0b01011100,
                                  0b11111100,
                                  0b00010110);
        let board_mirror = Bitboard::new(0b00101000,
                                         0b00101011,
                                         0b00110010,
                                         0b01100101,
                                         0b10100101,
                                         0b00111010,
                                         0b00111111,
                                         0b01101000);
        assert_eq!(board.mirror_horizontal(), board_mirror);
    }

    #[test]
    fn test_flip()
    {
        let board = Bitboard::new(0b00010100,
                                  0b11010100,
                                  0b01001100,
                                  0b10100110,
                                  0b10100101,
                                  0b01011100,
                                  0b11111100,
                                  0b00010110);
        let board_flip = Bitboard::new(0b00010110,
                                       0b11111100,
                                       0b01011100,
                                       0b10100101,
                                       0b10100110,
                                       0b01001100,
                                       0b11010100,
                                       0b00010100);
        assert_eq!(board.flip_vertical(), board_flip);
    }

    #[test]
    fn test_positive_horizontal_ray()
    {
        let board = Bitboard::new(0b00010100,
                                  0b11010100,
                                  0b01001100,
                                  0b10100110,
                                  0b10100101,
                                  0b01011100,
                                  0b11111100,
                                  0b00010110);
        let piece = board.pieces()
                         .nth(14)
                         .unwrap();

        // Make sure we got the correct piece
        assert_eq!(piece, BitboardPiece::from_square(26));

        let ray = Bitboard::new(0b00000000,
                                0b00000000,
                                0b00000000,
                                0b00000000,
                                0b00111000,
                                0b00000000,
                                0b00000000,
                                0b00000000);
        assert_eq!(board.positive_horizontal_ray(piece), ray);
    }

    #[test]
    fn test_negative_horizontal_ray()
    {
        let board = Bitboard::new(0b00010100,
                                  0b11010100,
                                  0b01001100,
                                  0b10100110,
                                  0b10100101,
                                  0b01011100,
                                  0b11111100,
                                  0b00010110);
        let piece = board.pieces()
                         .nth(14)
                         .unwrap();

        // Make sure we got the correct piece
        assert_eq!(piece, BitboardPiece::from_square(26));

        let ray = Bitboard::new(0b00000000,
                                0b00000000,
                                0b00000000,
                                0b00000000,
                                0b00000011,
                                0b00000000,
                                0b00000000,
                                0b00000000);
        assert_eq!(board.negative_horizontal_ray(piece), ray);
    }

    #[test]
    fn test_horizontal_ray()
    {
        let board = Bitboard::new(0b00010100,
                                  0b11010100,
                                  0b01001100,
                                  0b10100110,
                                  0b10100101,
                                  0b01011100,
                                  0b11111100,
                                  0b00010110);
        let piece = board.pieces()
                         .nth(14)
                         .unwrap();

        // Make sure we got the correct piece
        assert_eq!(piece, BitboardPiece::from_square(26));

        let ray = Bitboard::new(0b00000000,
                                0b00000000,
                                0b00000000,
                                0b00000000,
                                0b00111011,
                                0b00000000,
                                0b00000000,
                                0b00000000);
        assert_eq!(board.horizontal_ray(piece), ray);
    }

    #[test]
    fn test_positive_vertical_ray()
    {
        let board = Bitboard::new(0b00010100,
                                  0b11010100,
                                  0b01001000,
                                  0b10100010,
                                  0b10100101,
                                  0b01011100,
                                  0b11111100,
                                  0b00010110);
        let piece = board.pieces()
                         .nth(14)
                         .unwrap();

        // Make sure we got the correct piece
        assert_eq!(piece, BitboardPiece::from_square(26));

        let ray = Bitboard::new(0b00000000,
                                0b00000100,
                                0b00000100,
                                0b00000100,
                                0b00000000,
                                0b00000000,
                                0b00000000,
                                0b00000000);
        assert_eq!(board.positive_vertical_ray(piece), ray);
    }

    #[test]
    fn test_negative_vertical_ray()
    {
        let board = Bitboard::new(0b00010100,
                                  0b11010100,
                                  0b01001000,
                                  0b10100010,
                                  0b10100101,
                                  0b01011000,
                                  0b11111000,
                                  0b00010110);
        let piece = board.pieces()
                         .nth(12)
                         .unwrap();

        // Make sure we got the correct piece
        assert_eq!(piece, BitboardPiece::from_square(26));

        let ray = Bitboard::new(0b00000000,
                                0b00000000,
                                0b00000000,
                                0b00000000,
                                0b00000000,
                                0b00000100,
                                0b00000100,
                                0b00000100);
        assert_eq!(board.negative_vertical_ray(piece), ray);
    }

    #[test]
    fn test_vertical_ray()
    {
        let board = Bitboard::new(0b00010100,
                                  0b11010100,
                                  0b01001000,
                                  0b10100010,
                                  0b10100101,
                                  0b01011000,
                                  0b11111000,
                                  0b00010110);
        let piece = board.pieces()
                         .nth(12)
                         .unwrap();

        // Make sure we got the correct piece
        assert_eq!(piece, BitboardPiece::from_square(26));

        let ray = Bitboard::new(0b00000000,
                                0b00000100,
                                0b00000100,
                                0b00000100,
                                0b00000000,
                                0b00000100,
                                0b00000100,
                                0b00000100);
        assert_eq!(board.vertical_ray(piece), ray);
    }

    #[test]
    fn test_positive_diagonal_ray()
    {
        let board = Bitboard::new(0b00010100,
                                  0b11010100,
                                  0b01001000,
                                  0b10100010,
                                  0b10100101,
                                  0b01011000,
                                  0b11111000,
                                  0b00010110);
        let piece = board.pieces()
                         .nth(12)
                         .unwrap();

        // Make sure we got the correct piece
        assert_eq!(piece, BitboardPiece::from_square(26));

        let ray = Bitboard::new(0b01000000,
                                0b00100000,
                                0b00010000,
                                0b00001000,
                                0b00000000,
                                0b00000000,
                                0b00000000,
                                0b00000000);
        assert_eq!(board.positive_diagonal_ray(piece), ray);
    }

    #[test]
    fn test_negative_diagonal_ray()
    {
        let board = Bitboard::new(0b00010100,
                                  0b11010100,
                                  0b01001000,
                                  0b10100010,
                                  0b10100101,
                                  0b01011000,
                                  0b11111000,
                                  0b00010110);
        let piece = board.pieces()
                         .nth(12)
                         .unwrap();

        // Make sure we got the correct piece
        assert_eq!(piece, BitboardPiece::from_square(26));

        let ray = Bitboard::new(0b00000000,
                                0b00000000,
                                0b00000000,
                                0b00000000,
                                0b00000000,
                                0b00000010,
                                0b00000001,
                                0b00000000);
        assert_eq!(board.negative_diagonal_ray(piece), ray);
    }

    #[test]
    fn test_diagonal_ray()
    {
        let board = Bitboard::new(0b00010100,
                                  0b11010100,
                                  0b01001000,
                                  0b10100010,
                                  0b10100101,
                                  0b01011000,
                                  0b11111000,
                                  0b00010110);
        let piece = board.pieces()
                         .nth(12)
                         .unwrap();

        // Make sure we got the correct piece
        assert_eq!(piece, BitboardPiece::from_square(26));

        let ray = Bitboard::new(0b01000000,
                                0b00100000,
                                0b00010000,
                                0b00001000,
                                0b00000000,
                                0b00000010,
                                0b00000001,
                                0b00000000);
        assert_eq!(board.diagonal_ray(piece), ray);
    }

    #[test]
    fn test_positive_anti_diagonal_ray()
    {
        let board = Bitboard::new(0b00010100,
                                  0b11010100,
                                  0b01001000,
                                  0b10100010,
                                  0b10100101,
                                  0b01010000,
                                  0b11111000,
                                  0b00010110);
        let piece = board.pieces()
                         .nth(11)
                         .unwrap();

        // Make sure we got the correct piece
        assert_eq!(piece, BitboardPiece::from_square(26));

        let ray = Bitboard::new(0b00000000,
                                0b00000000,
                                0b00000000,
                                0b00000010,
                                0b00000000,
                                0b00000000,
                                0b00000000,
                                0b00000000);
        assert_eq!(board.positive_anti_diagonal_ray(piece), ray);
    }

    #[test]
    fn test_negative_anti_diagonal_ray()
    {
        let board = Bitboard::new(0b00010100,
                                  0b11010100,
                                  0b01001000,
                                  0b10100010,
                                  0b10100101,
                                  0b01010000,
                                  0b11111000,
                                  0b00010110);
        let piece = board.pieces()
                         .nth(11)
                         .unwrap();

        // Make sure we got the correct piece
        assert_eq!(piece, BitboardPiece::from_square(26));

        let ray = Bitboard::new(0b00000000,
                                0b00000000,
                                0b00000000,
                                0b00000000,
                                0b00000000,
                                0b00001000,
                                0b00010000,
                                0b00000000);
        assert_eq!(board.negative_anti_diagonal_ray(piece), ray);
    }

    #[test]
    fn test_anti_diagonal_ray()
    {
        let board = Bitboard::new(0b00010100,
                                  0b11010100,
                                  0b01001000,
                                  0b10100010,
                                  0b10100101,
                                  0b01010000,
                                  0b11111000,
                                  0b00010110);
        let piece = board.pieces()
                         .nth(11)
                         .unwrap();

        // Make sure we got the correct piece
        assert_eq!(piece, BitboardPiece::from_square(26));

        let ray = Bitboard::new(0b00000000,
                                0b00000000,
                                0b00000000,
                                0b00000010,
                                0b00000000,
                                0b00001000,
                                0b00010000,
                                0b00000000);
        assert_eq!(board.anti_diagonal_ray(piece), ray);
    }
}
