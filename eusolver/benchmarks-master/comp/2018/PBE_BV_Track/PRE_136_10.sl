
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


(constraint (= (f #xc8960d58101e6361) #x912c1ab0203cc6c2))
(constraint (= (f #x3c49a54b6653dc62) #xfffffffffffffffe))
(constraint (= (f #xc9c8200106ea8c53) #x939040020dd518a6))
(constraint (= (f #xd8da1cbe597682e3) #xb1b4397cb2ed05c6))
(constraint (= (f #x310a69e352cddc99) #x6214d3c6a59bb932))
(constraint (= (f #x5e284a0ebdeb0b4e) #xfffffffffffffffe))
(constraint (= (f #xd46d496623aa034c) #x95c95b4cee2afe58))
(constraint (= (f #x4003196eaa59981e) #xfffffffffffffffe))
(constraint (= (f #xb3a2ab8176a2e1ea) #xa62eaa3f44ae8f0b))
(constraint (= (f #xeca576ba37527a05) #xd94aed746ea4f40a))
(check-synth)
