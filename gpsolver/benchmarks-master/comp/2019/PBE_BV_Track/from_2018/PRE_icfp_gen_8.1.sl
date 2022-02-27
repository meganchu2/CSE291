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

(constraint (= (f #x5B322BB30C4E7A0D) #x0042002221084E41))
(constraint (= (f #xD0DE97A93CEA4A4F) #x0010D295211C4849))
(constraint (= (f #xBAA51FA109DE23EB) #x0010A1142109C422))
(constraint (= (f #x9ECAE85357ACC139) #x0098480842558802))
(constraint (= (f #x4877DEEAC4AA6969) #x000873DC4884082A))
(constraint (= (f #xCD42382DC1C53E85) #x0088420028008511))
(constraint (= (f #x1CDFE3F2AD29ABDB) #x0018DC6250A5212C))
(constraint (= (f #xDDA9B9769063A64F) #x0095212852006081))
(constraint (= (f #x643BDC85F447F5C3) #x00043B90848046B1))
(constraint (= (f #x5F3077702CB34A29) #x0046006600042141))
(constraint (= (f #xB8512B49BB7CFB96) #x00000B8512B49BB7))
(constraint (= (f #x510A1DB13814FA26) #x00000510A1DB1381))
(constraint (= (f #x590155584A8E2024) #x00000590155584A8))
(constraint (= (f #x7B37AA8CF336E118) #x000007B37AA8CF33))
(constraint (= (f #x0C7CF5B02864BBCC) #x000000C7CF5B0286))
(constraint (= (f #x210B74BE76E99192) #x0000000000000000))
(constraint (= (f #x97F1990599D925B0) #x0000000000000000))
(constraint (= (f #x57F1BCB5BF413B24) #x0000000000000000))
(constraint (= (f #x9945F7D188EB7680) #x0000000000000000))
(constraint (= (f #xEA34D4081BCB5724) #x0000000000000000))

(check-synth)

