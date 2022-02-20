(set-logic BV)

(define-fun ehad ((x (BitVec 64))) (BitVec 64)
    (bvlshr x #x0000000000000001))
(define-fun arba ((x (BitVec 64))) (BitVec 64)
    (bvlshr x #x0000000000000004))
(define-fun shesh ((x (BitVec 64))) (BitVec 64)
    (bvlshr x #x0000000000000010))
(define-fun smol ((x (BitVec 64))) (BitVec 64)
    (bvshl x #x0000000000000001))
(define-fun im ((x (BitVec 64)) (y (BitVec 64)) (z (BitVec 64))) (BitVec 64)
    (ite (= x #x0000000000000001) y z))
(synth-fun f ((x (BitVec 64))) (BitVec 64)
    ((Start (BitVec 64) (#x0000000000000000 #x0000000000000001 x (bvnot Start) (smol Start) (ehad Start) (arba Start) (shesh Start) (bvand Start Start) (bvor Start Start) (bvxor Start Start) (bvadd Start Start) (im Start Start Start)))))

(constraint (= (f #xB4872BEADD7B82C2) #x0000000000000002))
(constraint (= (f #xAEC8AF629ED85482) #x0000000000000002))
(constraint (= (f #xECF7A29AA0DBB66E) #x0000000000000002))
(constraint (= (f #x4F0117230A2329E0) #x0000000000000002))
(constraint (= (f #x4C28F26949A6CB44) #x0000000000000002))
(constraint (= (f #x6FCC3287BC9B0D38) #x0000000000000002))
(constraint (= (f #x6800107C6D95A75E) #x0000000000000002))
(constraint (= (f #x84F4C82BE44983F4) #x0000000000000002))
(constraint (= (f #xA323138F7B57FE98) #x0000000000000002))
(constraint (= (f #x42070E0F7AEEB2FE) #x0000000000000002))
(constraint (= (f #xCBC6C9B274E202EB) #x0000000000000002))
(constraint (= (f #x2FA81EBCD1F48841) #x0000000000000002))
(constraint (= (f #xC0D93B696189D6E9) #x0000000000000002))
(constraint (= (f #x51D38173913D7367) #x0000000000000002))
(constraint (= (f #xF21A16E12880C949) #x0000000000000002))
(constraint (= (f #x86C908EF1C523C37) #x000086C908EF1C53))
(constraint (= (f #x4E601FDEC6D90051) #x00004E601FDEC6D8))
(constraint (= (f #x8A556061D2CA6A13) #x00008A556061D2CB))
(constraint (= (f #x0087FCAAE38E2CF7) #x00000087FCAAE38F))
(constraint (= (f #xD8C56C660B41AF59) #x0000D8C56C660B40))
(constraint (= (f #x0000000000000001) #x0000000000000002))
(constraint (= (f #x841F786E4F10923A) #x0000000000000002))
(constraint (= (f #xDC01C6EEED38C61C) #x0000000000000002))
(constraint (= (f #xCE5E06FE6196FA13) #x0000CE5E06FE6197))
(constraint (= (f #x1D1A3D62116CF07C) #x0000000000000002))
(constraint (= (f #x0096F3658DB74ADA) #x0000000000000002))
(constraint (= (f #x8D0BF8C05E63800F) #x0000000000000002))
(constraint (= (f #xC627473B90267137) #x0000C627473B9027))
(constraint (= (f #x0628F21C3DA53074) #x0000000000000002))
(constraint (= (f #x00DA7D02BD2E420A) #x0000000000000002))
(constraint (= (f #xAC1F56C7A5734BC2) #x0000000000000002))
(constraint (= (f #x3FB615A0BF09399D) #x00003FB615A0BF08))
(constraint (= (f #x5EC348825B0EA1C3) #x0000000000000002))
(constraint (= (f #x3A1303321A3D6F13) #x00003A1303321A3C))

(check-synth)

