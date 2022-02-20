
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


(constraint (= (f #x31b89a6e3356b240) #xcb2bdbeae973e29b))
(constraint (= (f #xc192ea128ad82dd6) #xc192ea128ad82dd6))
(constraint (= (f #xe7a658ee0e27e583) #x09df418310f59c24))
(constraint (= (f #xee201d97438486a5) #xee201d97438486a4))
(constraint (= (f #xa85b91667db28e15) #x4d1eb5831a724909))
(constraint (= (f #xaa7321ab163764bd) #x4ae5ac3a386524f7))
(constraint (= (f #xdaebb8b1bbedeeec) #xdaebb8b1bbedeeec))
(constraint (= (f #x38a50903364a8d82) #xc3d0a66c9650c9a5))
(constraint (= (f #xeee47814c1d81989) #xeee47814c1d81988))
(constraint (= (f #xe06e034804a598cd) #xe06e034804a598cc))
(check-synth)
