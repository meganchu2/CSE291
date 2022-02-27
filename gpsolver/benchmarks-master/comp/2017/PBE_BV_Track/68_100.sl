
(set-logic BV)

(define-fun shr1 ((x (BitVec 64))) (BitVec 64) (bvlshr x #x0000000000000001))
(define-fun shr4 ((x (BitVec 64))) (BitVec 64) (bvlshr x #x0000000000000004))
(define-fun shr16 ((x (BitVec 64))) (BitVec 64) (bvlshr x #x0000000000000010))
(define-fun shl1 ((x (BitVec 64))) (BitVec 64) (bvshl x #x0000000000000001))
(define-fun if0 ((x (BitVec 64)) (y (BitVec 64)) (z (BitVec 64))) (BitVec 64) (ite (= x #x0000000000000001) y z))

(synth-fun f ( (x (BitVec 64))) (BitVec 64)
(

(Start (BitVec 64) (#x0000000000000000 #x0000000000000001 x (bvnot Start)
                    (shl1 Start)
 		    (shr1 Start)
		    (shr4 Start)
		    (shr16 Start)
		    (bvand Start Start)
		    (bvor Start Start)
		    (bvxor Start Start)
		    (bvadd Start Start)
		    (if0 Start Start Start)
 ))
)
)


(constraint (= (f #x0b6e9c2b8dd53c44) #x00000b6e9c2b8dd5))
(constraint (= (f #x3487eceeae7ead95) #x03487eceeae7ead9))
(constraint (= (f #xd7884d9d0b513187) #x0d7884d9d0b51318))
(constraint (= (f #x3101e035076d3221) #x00003101e035076d))
(constraint (= (f #x8852e06ab3564ea6) #x08852e06ab3564ea))
(constraint (= (f #x1eaea5ec4e264ae1) #x00001eaea5ec4e26))
(constraint (= (f #x8eb2eec5b1e785cc) #x00008eb2eec5b1e7))
(constraint (= (f #x6d348abedbce8c50) #x06d348abedbce8c5))
(constraint (= (f #x4e2594643dce5c9e) #x00004e2594643dce))
(constraint (= (f #x6e1e87d57c2ae697) #x00006e1e87d57c2a))
(constraint (= (f #x8ec48e20a5aebca6) #x08ec48e20a5aebca))
(constraint (= (f #xa1e276eb5e39e79d) #x0a1e276eb5e39e79))
(constraint (= (f #x6adc2c495101cea1) #x00006adc2c495101))
(constraint (= (f #xbd3201ae9ee1dcae) #x0bd3201ae9ee1dca))
(constraint (= (f #x24ec348e82cb144e) #x024ec348e82cb144))
(constraint (= (f #x8598725eb3086eac) #x00008598725eb308))
(constraint (= (f #xcc0a6aad7b46b5a8) #x0000cc0a6aad7b46))
(constraint (= (f #x05ea6ecb3d966201) #x000005ea6ecb3d96))
(constraint (= (f #x920320e448b9c5bd) #x0920320e448b9c5b))
(constraint (= (f #x14e54b05a3b29513) #x000014e54b05a3b2))
(constraint (= (f #xd4b26c1ba5e3e2ee) #x0d4b26c1ba5e3e2e))
(constraint (= (f #x92e131041a6eb144) #x000092e131041a6e))
(constraint (= (f #x6a0a730c54934008) #x00006a0a730c5493))
(constraint (= (f #xb97b0286475a1911) #x0b97b0286475a191))
(constraint (= (f #x430eadd3b6de983d) #x0430eadd3b6de983))
(constraint (= (f #xed594342de2c237d) #x0ed594342de2c237))
(constraint (= (f #x1ea3ebb9e9c98913) #x00001ea3ebb9e9c9))
(constraint (= (f #xb14e819aa7d6beb0) #x0b14e819aa7d6beb))
(constraint (= (f #x1c9533e9e469c16d) #x00001c9533e9e469))
(constraint (= (f #x47c385161eb5b0d4) #x047c385161eb5b0d))
(constraint (= (f #xb5849e0baa4372a8) #x0000b5849e0baa43))
(constraint (= (f #xe7289022e358ee57) #x0000e7289022e358))
(constraint (= (f #x0dbcaa052adbe612) #x00000dbcaa052adb))
(constraint (= (f #xdc0162dc9e75e4e3) #x0dc0162dc9e75e4e))
(constraint (= (f #x16ac971d2e411524) #x000016ac971d2e41))
(constraint (= (f #x19e3d6aead3dc65c) #x019e3d6aead3dc65))
(constraint (= (f #x3808be4976a16ea6) #x03808be4976a16ea))
(constraint (= (f #xe44ee87ae9e184a7) #x0e44ee87ae9e184a))
(constraint (= (f #xbc37421a0253a828) #x0000bc37421a0253))
(constraint (= (f #x29e409e2471137e3) #x029e409e2471137e))
(constraint (= (f #x0884e17bcc7638eb) #x00884e17bcc7638e))
(constraint (= (f #x9b593969630dea18) #x09b593969630dea1))
(constraint (= (f #x4a0ebb23195eebe7) #x04a0ebb23195eebe))
(constraint (= (f #x9582a75302c3e960) #x00009582a75302c3))
(constraint (= (f #x6c94865b22deee96) #x00006c94865b22de))
(constraint (= (f #x85e38e26876c83e3) #x085e38e26876c83e))
(constraint (= (f #x2eab535a83476b24) #x00002eab535a8347))
(constraint (= (f #x670b66a4da3c21c8) #x0000670b66a4da3c))
(constraint (= (f #xb8e9c8796eca773e) #x0000b8e9c8796eca))
(constraint (= (f #x7de98ec4a41ce489) #x00007de98ec4a41c))
(constraint (= (f #x848e46e85231d57d) #x0848e46e85231d57))
(constraint (= (f #xe7ea9b5ce61bd633) #x0000e7ea9b5ce61b))
(constraint (= (f #x930dc3503c4dd6e7) #x0930dc3503c4dd6e))
(constraint (= (f #x24b2e551aee09e89) #x000024b2e551aee0))
(constraint (= (f #x78e819a51aeee9bb) #x000078e819a51aee))
(constraint (= (f #x986e9530cebee39d) #x0986e9530cebee39))
(constraint (= (f #xe01ca22514ed4ee7) #x0e01ca22514ed4ee))
(constraint (= (f #xeb77c50c0e051d9e) #x0000eb77c50c0e05))
(constraint (= (f #x75b92bc9e5b8e202) #x075b92bc9e5b8e20))
(constraint (= (f #x34e281aeab9de1a5) #x000034e281aeab9d))
(constraint (= (f #x328d0eee1e7495ae) #x0328d0eee1e7495a))
(constraint (= (f #xabd8eb469da23e5b) #x0000abd8eb469da2))
(constraint (= (f #x44aabcd70a4e12dc) #x044aabcd70a4e12d))
(constraint (= (f #x8ea0056157044b0e) #x08ea0056157044b0))
(constraint (= (f #xe99a25d455ea4aee) #x0e99a25d455ea4ae))
(constraint (= (f #x735ada472a35bab4) #x0735ada472a35bab))
(constraint (= (f #xcbdb99a8ee97e560) #x0000cbdb99a8ee97))
(constraint (= (f #xb09714e098346e4e) #x0b09714e098346e4))
(constraint (= (f #xe0dec9ea4a038528) #x0000e0dec9ea4a03))
(constraint (= (f #xba4e60821c378039) #x0ba4e60821c37803))
(constraint (= (f #xc8697e3c981a3a2a) #x0c8697e3c981a3a2))
(constraint (= (f #x0e9d4638e9d964c6) #x00e9d4638e9d964c))
(constraint (= (f #x37eae1ce30731587) #x037eae1ce3073158))
(constraint (= (f #x14ee9452d002c61e) #x000014ee9452d002))
(constraint (= (f #x8e3e642d061d7dd0) #x08e3e642d061d7dd))
(constraint (= (f #x8d6bc5884e5871be) #x00008d6bc5884e58))
(constraint (= (f #xe7e1d95d7149c96d) #x0000e7e1d95d7149))
(constraint (= (f #xb1457a00bed1eedd) #x0b1457a00bed1eed))
(constraint (= (f #x6a4130a64e3baec3) #x06a4130a64e3baec))
(constraint (= (f #xc4e83452b73a2d65) #x0000c4e83452b73a))
(constraint (= (f #xadad0ba2eec87458) #x0adad0ba2eec8745))
(constraint (= (f #x9d93d807b54c7aea) #x09d93d807b54c7ae))
(constraint (= (f #x9eb4abd2c4da1630) #x09eb4abd2c4da163))
(constraint (= (f #x7e614dcbb0c0a335) #x07e614dcbb0c0a33))
(constraint (= (f #xb320a989cab5b15c) #x0b320a989cab5b15))
(constraint (= (f #x3583b1aeace96ebc) #x03583b1aeace96eb))
(constraint (= (f #x5664b04a73ce70b4) #x05664b04a73ce70b))
(constraint (= (f #xb434ec03c0934e0e) #x0b434ec03c0934e0))
(constraint (= (f #x2d5897d8c617d993) #x00002d5897d8c617))
(constraint (= (f #x5893a60a6ee303e3) #x05893a60a6ee303e))
(constraint (= (f #x9e766e56ea0ae826) #x09e766e56ea0ae82))
(constraint (= (f #x618acee3869060e9) #x0000618acee38690))
(constraint (= (f #x5bad7976e9c0c611) #x05bad7976e9c0c61))
(constraint (= (f #xe024657e69a23479) #x0e024657e69a2347))
(constraint (= (f #x3229332ea1be1ddd) #x03229332ea1be1dd))
(constraint (= (f #x35e4ce4ce8491b48) #x000035e4ce4ce849))
(constraint (= (f #x0926ee0ee3330ea3) #x00926ee0ee3330ea))
(constraint (= (f #x26ece37d3e43e457) #x000026ece37d3e43))
(constraint (= (f #x83e27a4a6e415d6c) #x000083e27a4a6e41))
(constraint (= (f #x56444c7330824456) #x000056444c733082))
(check-synth)
