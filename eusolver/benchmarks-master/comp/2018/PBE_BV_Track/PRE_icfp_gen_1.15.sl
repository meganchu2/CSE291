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
(constraint (= (f #x3335CFB6199F07E5) #x0000000000000002))
(constraint (= (f #xA4E09AE8529CBCAE) #x0000000000000002))
(constraint (= (f #x1081268BF5554867) #x0000000000000002))
(constraint (= (f #x7DCB54C972D04832) #x0000000000000002))
(constraint (= (f #x4ADFB9B79C3D4D7D) #x0000000000000002))
(constraint (= (f #x3791F1E2308F32B4) #x3791F1E2308F32B4))
(constraint (= (f #x8E012643536B2A6C) #x8E012643536B2A6C))
(constraint (= (f #xA96B206A78C9C7BE) #xA96B206A78C9C7BE))
(constraint (= (f #xA16175DAAC05D24E) #xA16175DAAC05D24E))
(constraint (= (f #xA9DB6CFCEA60A468) #xA9DB6CFCEA60A468))
(constraint (= (f #xF0F0F0F0F0F0F0F2) #x0000000000000002))
(constraint (= (f #xAAAAAAAAAAAAAAAB) #xAAAAAAAAAAAAAAAA))
(constraint (= (f #x0000000000000001) #x0000000000000000))
(constraint (= (f #xA04026321AC588CA) #xA04026321AC588CA))
(constraint (= (f #xADCD34F1FE4EA326) #xADCD34F1FE4EA326))
(constraint (= (f #x9B968444F55F56C0) #x0000000000000002))
(constraint (= (f #x781248C9978CFFB3) #x781248C9978CFFB2))
(constraint (= (f #x95CEB97EF4A0FE25) #x95CEB97EF4A0FE24))
(constraint (= (f #xF47597868CECBCED) #xF47597868CECBCEC))
(constraint (= (f #x2CD25E455673C463) #x0000000000000002))
(constraint (= (f #x91DF2D16D35C9552) #x0000000000000002))
(constraint (= (f #xA80BCB8886379C50) #x0000000000000002))
(constraint (= (f #xC9E01855476CE422) #xC9E01855476CE422))
(constraint (= (f #x0000000000000001) #x0000000000000000))
(constraint (= (f #xAAAAAAAAAAAAAAAB) #xAAAAAAAAAAAAAAAA))
(constraint (= (f #x4A948A112A529293) #x0000000000000002))
(constraint (= (f #x4AA4204A92548083) #x0000000000000002))
(constraint (= (f #x5EEFBBC18DE4AFC8) #x5EEFBBC18DE4AFC8))
(constraint (= (f #xD1F293C89B72B754) #x0000000000000002))
(constraint (= (f #xCE890B1932C7AB8D) #xCE890B1932C7AB8C))
(constraint (= (f #xF0F0F0F0F0F0F0F2) #x0000000000000002))
(constraint (= (f #x15B6FD1316945FF8) #x0000000000000002))
(constraint (= (f #x76527C1B23D057C9) #x0000000000000002))
(constraint (= (f #x7AEDE5B7270743F2) #x7AEDE5B7270743F2))
(constraint (= (f #xA08F1C4E676D3603) #xA08F1C4E676D3602))
(check-synth)
