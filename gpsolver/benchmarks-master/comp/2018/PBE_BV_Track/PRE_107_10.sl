
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


(constraint (= (f #xc7c78c355c3eb4ec) #x7070e79547829624))
(constraint (= (f #x582b3633a28793e4) #x0000582b3633a287))
(constraint (= (f #xacad5b76053e37ee) #xa6a54913f5839020))
(constraint (= (f #x889551ee25b285b7) #xeed55c23b49af492))
(constraint (= (f #x53cd253676e37e4a) #x000053cd253676e3))
(constraint (= (f #x412a7694e0b8dcc2) #x7dab12d63e8e4678))
(constraint (= (f #xd5c973a98792891c) #x546d18acf0daedc4))
(constraint (= (f #xb773418550a85c3e) #x91197cf55eaf4780))
(constraint (= (f #xe0ee505c5617d429) #x0000e0ef314aa674))
(constraint (= (f #x127ce8d7eaae1bcc) #xdb062e502aa3c864))
(check-synth)
