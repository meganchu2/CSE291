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

(constraint (= (f #x09EE2CC0649C2A23) #x09EE2CC0649C2A22))
(constraint (= (f #x40A8B1E6834857D7) #x40A8B1E6834857D6))
(constraint (= (f #xA60EB9B05B70E406) #xA60EB9B05B70E407))
(constraint (= (f #x0B5722B550220243) #x0B5722B550220242))
(constraint (= (f #x909B5BEE2EF64881) #x909B5BEE2EF64880))
(constraint (= (f #x0000000000C45C04) #x0000000000C45C05))
(constraint (= (f #x00000000007CFD28) #x00000000007CFD29))
(constraint (= (f #x00000000009EE90C) #x00000000009EE90D))
(constraint (= (f #x00000000007A4C7D) #x00000000007A4C7C))
(constraint (= (f #x0000000000D44850) #x0000000000D44851))
(constraint (= (f #x29B9DA2B04892D7B) #x06B2312EA7DBB694))
(constraint (= (f #x6117E56D91312182) #x04F740D4937676F3))
(constraint (= (f #x5292C0F2335BA0C8) #x056B69F86E6522F9))
(constraint (= (f #xAACCDAC37619BD7A) #x02A99929E44F3214))
(constraint (= (f #xEDAE97B7B7D79DD0) #x00928B4242414311))
(constraint (= (f #x0000000000000001) #x0000000000000000))
(constraint (= (f #x0000000000C36726) #x0000000000C36727))
(constraint (= (f #x0000000000DFC732) #x0000000000DFC733))
(constraint (= (f #x0000000000C5BDE3) #x0000000000C5BDE2))
(constraint (= (f #x0000000000BDB1AF) #x0000000000BDB1AE))
(constraint (= (f #x0000000000CF80D5) #x0000000000CF80D4))
(constraint (= (f #xFFFFFFFFFFFF8378) #x07FFFFFFFFFFFFFF))
(constraint (= (f #xFFFFFFFFFFFF283A) #x07FFFFFFFFFFFFFF))
(constraint (= (f #xFFFFFFFFFFFF1CB8) #x07FFFFFFFFFFFFFF))
(constraint (= (f #xFFFFFFFFFFFFC0E9) #x07FFFFFFFFFFFFFF))
(constraint (= (f #xFFFFFFFFFFFF6ADE) #x07FFFFFFFFFFFFFF))
(constraint (= (f #xFFFF0000FFFF0002) #x000007FFF80007FF))

(check-synth)

