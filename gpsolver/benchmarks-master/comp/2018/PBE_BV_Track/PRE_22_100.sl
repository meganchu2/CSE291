
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


(constraint (= (f #x348a21ba5100d534) #x348a21ba5100d534))
(constraint (= (f #x464115d04870ce0a) #x464115d04870ce0a))
(constraint (= (f #x5ecbae527c067106) #x5ecbae527c067106))
(constraint (= (f #xb29d31896d0a3616) #xb29d31896d0a3616))
(constraint (= (f #x8e6975de5768d27e) #x8e6975de5768d27e))
(constraint (= (f #x5497db0522082214) #x5497db0522082214))
(constraint (= (f #x5e3b8826eea0b648) #x5e3b8826eea0b648))
(constraint (= (f #x1247e5cc49b769d1) #x5b677cfd70951115))
(constraint (= (f #x6c5cad64be23c43b) #x1dcf62f7b6b2d527))
(constraint (= (f #xcb2e932e6de4d0d6) #xcb2e932e6de4d0d6))
(constraint (= (f #x0162b720d5a38266) #x0162b720d5a38266))
(constraint (= (f #x87750863c14e7b12) #x87750863c14e7b12))
(constraint (= (f #x55150ceb000e47a7) #xa969409700476643))
(constraint (= (f #x3889542231a558e8) #x3889542231a558e8))
(constraint (= (f #x1247289058cae1de) #x1247289058cae1de))
(constraint (= (f #xea4608d2869ce52c) #xea4608d2869ce52c))
(constraint (= (f #x8aea4ebdb933819e) #x8aea4ebdb933819e))
(constraint (= (f #x0eed1a0c4dc917ac) #x0eed1a0c4dc917ac))
(constraint (= (f #x60d74ea33e55e09a) #x60d74ea33e55e09a))
(constraint (= (f #xd1a3b5d1e5113b99) #x18328d19795629fd))
(constraint (= (f #x4b56ed683e871521) #x78b2a30938a369a5))
(constraint (= (f #x65ced3cc747096a0) #x65ced3cc747096a0))
(constraint (= (f #x6e755e050e5ee2a0) #x6e755e050e5ee2a0))
(constraint (= (f #x674363b2eebd5bc7) #x0450f27ea9b2cae3))
(constraint (= (f #xe097e54ee19881d9) #x62f77a8a67fa893d))
(constraint (= (f #x93e802015b13e015) #xe3880a06c7636069))
(constraint (= (f #xe56cba16292e5c1c) #xe56cba16292e5c1c))
(constraint (= (f #x1eae67eec5a89cee) #x1eae67eec5a89cee))
(constraint (= (f #xe7bd0568a0b347b1) #x86b11b0b23806675))
(constraint (= (f #xaa9e2a0a02b0515c) #xaa9e2a0a02b0515c))
(constraint (= (f #x84d8a4bd1c7cdcc0) #x84d8a4bd1c7cdcc0))
(constraint (= (f #xd6535947e5c2520c) #xd6535947e5c2520c))
(constraint (= (f #xadb5a6ee88b7ccd3) #x648c42a8ab97001f))
(constraint (= (f #x77b0edbeb778de3d) #x5674a4b9955c5731))
(constraint (= (f #xecd1098a4587259e) #xecd1098a4587259e))
(constraint (= (f #x28a39ca3417dd424) #x28a39ca3417dd424))
(constraint (= (f #x44ce0ae237e1c50e) #x44ce0ae237e1c50e))
(constraint (= (f #x82ad37e396e3e21d) #x8d621771f2736a91))
(constraint (= (f #x9b4dde8e3d254c44) #x9b4dde8e3d254c44))
(constraint (= (f #xe91810bd95e5ea99) #x8d7853b3ed7d94fd))
(constraint (= (f #x8e3ec537b1e0e96e) #x8e3ec537b1e0e96e))
(constraint (= (f #xe019c5102d01a900) #xe019c5102d01a900))
(constraint (= (f #x12ec446ece872e75) #x5e9d562a08a3e849))
(constraint (= (f #x2582cce208341a91) #xbb8e006a290484d5))
(constraint (= (f #xe3756b6e080ea0db) #x714b192628492447))
(constraint (= (f #xd576e607392c0b9e) #xd576e607392c0b9e))
(constraint (= (f #x737d38aee692c1ae) #x737d38aee692c1ae))
(constraint (= (f #x4b5220e9e3881a07) #x789aa49171a88223))
(constraint (= (f #xa4123b50154aa71a) #xa4123b50154aa71a))
(constraint (= (f #xa911494dce336ea0) #xa911494dce336ea0))
(constraint (= (f #x5b2a61eb03822b9c) #x5b2a61eb03822b9c))
(constraint (= (f #xcd6abccbc2048ede) #xcd6abccbc2048ede))
(constraint (= (f #xd0d07c427222ec5b) #x14126d4c3aae9dc7))
(constraint (= (f #xed2629e8e52e190e) #xed2629e8e52e190e))
(constraint (= (f #x86e72eb42705de3c) #x86e72eb42705de3c))
(constraint (= (f #xa6476beb4c46e188) #xa6476beb4c46e188))
(constraint (= (f #xae696266680655ce) #xae696266680655ce))
(constraint (= (f #xd78cb4e7d6b79ca3) #x35bf888731960f2f))
(constraint (= (f #x889606d0aeace617) #xaaee221369607e73))
(constraint (= (f #xe012278a8b8b0b2e) #xe012278a8b8b0b2e))
(constraint (= (f #x5a27dce242662be0) #x5a27dce242662be0))
(constraint (= (f #xb809eb5c9d2d7240) #xb809eb5c9d2d7240))
(constraint (= (f #x91e496d189ad5e90) #x91e496d189ad5e90))
(constraint (= (f #xc0597d0e357c2550) #xc0597d0e357c2550))
(constraint (= (f #xde2937ed19e68373) #x56ce17a18180913f))
(constraint (= (f #x94c2db3b5b55da6a) #x94c2db3b5b55da6a))
(constraint (= (f #x41c2d0104836ae38) #x41c2d0104836ae38))
(constraint (= (f #x78975160a417415d) #x5af496e3347446d1))
(constraint (= (f #xe47b5e9ce3e54887) #x7668d910737a6aa3))
(constraint (= (f #xcbeeae10cc9b0dbe) #xcbeeae10cc9b0dbe))
(constraint (= (f #xc8440ca5e3ac4dda) #xc8440ca5e3ac4dda))
(constraint (= (f #xe9b6136ea0e9709c) #xe9b6136ea0e9709c))
(constraint (= (f #x6decd02ee1983880) #x6decd02ee1983880))
(constraint (= (f #xa873405d8d6ee57b) #x4a4041d3c32a7b67))
(constraint (= (f #x1a7e25d18d12d199) #x8476bd17c15e17fd))
(constraint (= (f #xb39d2372c3a2e56a) #xb39d2372c3a2e56a))
(constraint (= (f #xdc8b08a0da18b1d0) #xdc8b08a0da18b1d0))
(constraint (= (f #x621535c0925e9063) #xea6a0cc2dbd8d1ef))
(constraint (= (f #x3d78864e7291aae6) #x3d78864e7291aae6))
(constraint (= (f #x82a517ecbe10637e) #x82a517ecbe10637e))
(constraint (= (f #x3acad6835cbdae4d) #x25f63090cfb46781))
(constraint (= (f #x73d51e92e7e027dc) #x73d51e92e7e027dc))
(constraint (= (f #xe10a9e4a14a0979d) #x653517726722f611))
(constraint (= (f #xeb026989b9146857) #x970c0fb09d6609b3))
(constraint (= (f #x6879e9a9d29ce87e) #x6879e9a9d29ce87e))
(constraint (= (f #x2cc1a978ca953998) #x2cc1a978ca953998))
(constraint (= (f #xe9bad48db0390123) #x90a626c4711d05af))
(constraint (= (f #xe02ea69ec1bce7eb) #x60e94119c8b08797))
(constraint (= (f #xa3ae92c7499be899) #x3268dde4700b8afd))
(constraint (= (f #x031e619bee60a290) #x031e619bee60a290))
(constraint (= (f #xb9b41b84a87e5ec7) #xa08489974a77d9e3))
(constraint (= (f #xd07ee2a6534ecd72) #xd07ee2a6534ecd72))
(constraint (= (f #x929c85b29c928eee) #x929c85b29c928eee))
(constraint (= (f #xaab61556207bb0b9) #x558e6aaea26a739d))
(constraint (= (f #xd6aa9813044ad908) #xd6aa9813044ad908))
(constraint (= (f #x8ec274d68dc23c8e) #x8ec274d68dc23c8e))
(constraint (= (f #x226749da9ecd6b11) #xac0471451a031755))
(constraint (= (f #xd6ce2567e7ab9d39) #x3206bb07865a121d))
(constraint (= (f #x3dc0806e5c09c4ed) #x34c28227cc30d8a1))
(constraint (= (f #x39247ede591b12d4) #x39247ede591b12d4))
(check-synth)
