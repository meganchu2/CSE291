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
(constraint (= (f #xEC74B3433A481A65) #xEC74FF77BB4B3A6D))
(constraint (= (f #x67FCD2F1323CD0C7) #x67FCF7FDF2FDF2FF))
(constraint (= (f #x14D902E206CEB7CB) #x14D916FB06EEB7CF))
(constraint (= (f #x5A989F1545F2B66B) #x5A98DF9DDFF7F7FB))
(constraint (= (f #x18D3D06DAB106FA0) #x18D3D8FFFB7DEFB0))
(constraint (= (f #x0000001B254AFD8B) #x0000001B255BFDCB))
(constraint (= (f #x0000001381167A4C) #x000000138117FB5E))
(constraint (= (f #x00000019119FFDCD) #x00000019119FFDDF))
(constraint (= (f #x000000148201BE04) #x000000148215BE05))
(constraint (= (f #x00000015F4C2E46C) #x00000015F4D7F4EE))
(constraint (= (f #x0000000000000000) #x0000000000000000))
(constraint (= (f #xFFFFFFFFC0000002) #xFFFFFFFFFFFFC002))
(constraint (= (f #x0000000000000001) #x0000000000000001))
(constraint (= (f #xCD42E236F9FCEE76) #x0000006001701A74))
(constraint (= (f #x74BAE1C4F1DEC1F4) #x000000304070E260))
(constraint (= (f #x281135A3FC546270) #x00000010009A0030))
(constraint (= (f #x86C3CABD9389FF19) #x0000004140C144C9))
(constraint (= (f #x97A7554A142A6F73) #x0000000A810A0502))
(constraint (= (f #x0000001EB826B1D6) #x000000000F5C1358))
(constraint (= (f #x0000001E4861061C) #x000000000F243083))
(constraint (= (f #x0000001DA579CC7C) #x000000000ED2BCE6))
(constraint (= (f #x0000001ED791927E) #x000000000F6BC8C9))
(constraint (= (f #x00000016DB9534DE) #x000000000B6DCA9A))
(constraint (= (f #x5FA264119E94E1BB) #x0000002200020840))
(constraint (= (f #x8D98AE8E3B4A1EB3) #x000000464415050D))
(constraint (= (f #xB2983D7178143EF4) #x00000018081C081C))
(constraint (= (f #x3B66F8BFB83E0077) #x0000001C135C1F00))
(constraint (= (f #x0D4E6F7903077477) #x00000006A4018080))
(constraint (= (f #x537902FF99BBBDE1) #x537953FF9BFFBDFB))
(constraint (= (f #x81B4CB3D685D5778) #x000000409A240EA0))
(constraint (= (f #xF0245A0F9C2916BB) #x00000028020C048A))
(constraint (= (f #x7333B2486CC02A90) #x0000001900102014))
(constraint (= (f #xCB6636155CB8201E) #x00000001020A0800))
(constraint (= (f #x000000151DEDED37) #x000000000A8EF6F6))
(constraint (= (f #x0000000000000000) #x0000000000000000))
(constraint (= (f #x000000153BD82342) #x000000153BDD3BDA))
(constraint (= (f #x00000018BF0EEDD4) #x000000000C5F8776))
(constraint (= (f #x0000000000000001) #x0000000000000001))
(constraint (= (f #x99624CCB4DE43462) #x9962DDEB4DEF7DE6))
(check-synth)
