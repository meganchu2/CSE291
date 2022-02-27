
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


(constraint (= (f #x1dea8379eebdceb0) #xffffffffffffffff))
(constraint (= (f #x2abc048154191b93) #xfffffffffffffffe))
(constraint (= (f #xa8472bd3838ebdc8) #xaf71a858f8e2846f))
(constraint (= (f #x20ed136260d51b5a) #xbe25d93b3e55c94b))
(constraint (= (f #xe1ba389d81122339) #x3c8b8ec4fddbb98d))
(constraint (= (f #xed72e6e97c80ec49) #x251a322d06fe276d))
(constraint (= (f #x2bab98ddb4a69d3d) #xa8a8ce4496b2c585))
(constraint (= (f #x1ebb8ac38e903975) #xfffffffffffffffe))
(constraint (= (f #x6507809ee4ccee86) #xffffffffffffffff))
(constraint (= (f #x1a1b7cde82e79eb1) #xfffffffffffffffe))
(check-synth)
