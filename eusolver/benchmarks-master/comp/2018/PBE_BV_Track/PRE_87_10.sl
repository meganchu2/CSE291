
(set-logic BV)

(define-fun ehad ((x (BitVec 64))) (BitVec 64) (bvlshr x #x0000000000000001))
(define-fun arba ((x (BitVec 64))) (BitVec 64) (bvlshr x #x0000000000000004))
(define-fun shesh ((x (BitVec 64))) (BitVec 64) (bvlshr x #x0000000000000010))
(define-fun smol ((x (BitVec 64))) (BitVec 64) (bvshl x #x0000000000000001))
(define-fun im ((x (BitVec 64)) (y (BitVec 64)) (z (BitVec 64))) (BitVec 64) (ite (= x #x0000000000000001) y z))

(synth-fun f ( (x (BitVec 64))) (BitVec 64)
(

(Start (BitVec 64) (#x0000000000000000 #x0000000000000001 x (bvnot Start)
                    (smol Start)
 		    (ehad Start)
		    (arba Start)
		    (shesh Start)
		    (bvand Start Start)
		    (bvor Start Start)
		    (bvxor Start Start)
		    (bvadd Start Start)
		    (im Start Start Start)
 ))
)
)


(constraint (= (f #x66724b71e3b8a452) #x66724b71e3b8a454))
(constraint (= (f #x6b566e6d3670d9ce) #x6b566e6d3670d9d0))
(constraint (= (f #xd3e1ac6bb2e995c3) #x0000000000000002))
(constraint (= (f #x0226c8a04ebeea5e) #x0226c8a04ebeea60))
(constraint (= (f #x9489e05b83e0784c) #x9489e05b83e0784e))
(constraint (= (f #x87de2ed85dc94818) #x43ef176c2ee4a40c))
(constraint (= (f #x3352ea4b75c79e83) #x0000000000000002))
(constraint (= (f #x2970e37c57ad922e) #x14b871be2bd6c917))
(constraint (= (f #xbba2644d2de32e14) #x5dd1322696f1970a))
(constraint (= (f #xe481b69326876ec3) #x0000000000000002))
(check-synth)
