
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


(constraint (= (f #x4cb86ddc83ce50a2) #x4cb86ddc83ce50a2))
(constraint (= (f #xec64bb73d0e8ba14) #xec64bb73d0e8ba14))
(constraint (= (f #x7cae1d68e5ee2eb8) #x7cae1d68e5ee2eb8))
(constraint (= (f #x1aedd0e026c49408) #x1aedd0e026c49408))
(constraint (= (f #x540b2c9e007b5422) #x540b2c9e007b5422))
(constraint (= (f #x3ea34ed7052e99db) #x3ea34ed7052e99d9))
(constraint (= (f #x9900ed412c53262c) #x9900ed412c53262c))
(constraint (= (f #x8e21e59225eae682) #x8e21e59225eae682))
(constraint (= (f #x81bc9ed221c6a904) #x81bc9ed221c6a904))
(constraint (= (f #x12e6ec5aac0e57e7) #x12e6ec5aac0e57e5))
(check-synth)
