
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


(constraint (= (f #x4b44d589543a565b) #x04b44d589543a565))
(constraint (= (f #x60e4741ede572dae) #x060e4741ede572db))
(constraint (= (f #x377e80057ae8bca6) #x377e80057ae8bca6))
(constraint (= (f #xe5aa51edae79ee50) #x0e5aa51edae79ee5))
(constraint (= (f #x2c947e52d713642d) #x02c947e52d713642))
(constraint (= (f #xe0680844d6470e22) #xe0680844d6470e22))
(constraint (= (f #xe6ddc3ee7242a70c) #xe6ddc3ee7242a70c))
(constraint (= (f #x46bb5e5b69cdb903) #x46bb5e5b69cdb904))
(constraint (= (f #xba16321b911b1ead) #x0ba16321b911b1ea))
(constraint (= (f #x6987b7c65b0725ce) #x6987b7c65b0725ce))
(check-synth)
