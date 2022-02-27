(set-logic BV)

(define-fun ehad ((x (_ BitVec 64))) (_ BitVec 64)
    (bvlshr x #x0000000000000001))
(define-fun arba ((x (_ BitVec 64))) (_ BitVec 64)
    (bvlshr x #x0000000000000004))
(define-fun shesh ((x (_ BitVec 64))) (_ BitVec 64)
    (bvlshr x #x0000000000000010))
(define-fun smol ((x (_ BitVec 64))) (_ BitVec 64)
    (bvshl x #x0000000000000001))
(define-fun im ((x (_ BitVec 64)) (y (_ BitVec 64)) (z (_ BitVec 64))) (_ BitVec 64)
    (ite (= x #x0000000000000001) y z))
(synth-fun f ((x (_ BitVec 64))) (_ BitVec 64)
    ((Start (_ BitVec 64)))
    ((Start (_ BitVec 64) (#x0000000000000000 #x0000000000000001 x (bvnot Start) (smol Start) (ehad Start) (arba Start) (shesh Start) (bvand Start Start) (bvor Start Start) (bvxor Start Start) (bvadd Start Start) (im Start Start Start)))))

(constraint (= (f #x8e61c8adc9e754ae) #x00008e61c8adc9e7))
(constraint (= (f #x8415ece740b5a048) #x00008415ece740b5))
(constraint (= (f #x7395623ec16527ee) #x00007395623ec165))
(constraint (= (f #xee184aa6c112d2eb) #xdc30954d8225a5d8))
(constraint (= (f #x371eb0e48e476ed7) #x6e3d61c91c8eddb0))
(constraint (= (f #x1eee35c1000ddd04) #x00001eee35c1000d))
(constraint (= (f #x9ae588e7ed43e97d) #x35cb11cfda87d2fc))
(constraint (= (f #x9003aece8ee81aac) #x00009003aece8ee8))
(constraint (= (f #x1c6ebcb17eabcd91) #x38dd7962fd579b24))
(constraint (= (f #x30dd2a7490c61644) #x000030dd2a7490c6))
(constraint (= (f #x71c0eee859c4007c) #x000071c0eee859c4))
(constraint (= (f #xe3ce2e5e3e9ecd98) #x0000e3ce2e5e3e9e))
(constraint (= (f #xec7be484214c48bd) #xd8f7c9084298917c))
(constraint (= (f #x40e76345e1217169) #x81cec68bc242e2d4))
(constraint (= (f #x8de2c7d05c5e79b5) #x1bc58fa0b8bcf36c))
(constraint (= (f #x6474e00daae03028) #x00006474e00daae0))
(constraint (= (f #x57db133a2bed1b02) #x000057db133a2bed))
(constraint (= (f #x6d1a5c05c92335ca) #x00006d1a5c05c923))
(constraint (= (f #xd028ed47e450cd12) #x0000d028ed47e450))
(constraint (= (f #x33a5b94cbcb54159) #x674b7299796a82b4))
(constraint (= (f #xe7a7b900e058261a) #x0000e7a7b900e058))
(constraint (= (f #x26ad8cce7b4e6aac) #x000026ad8cce7b4e))
(constraint (= (f #x2b136722bb057b79) #x5626ce45760af6f4))
(constraint (= (f #x917e28d46b9eda9e) #x0000917e28d46b9e))
(constraint (= (f #x3799234dd6e334e2) #x00003799234dd6e3))
(constraint (= (f #x61a0bbc2b75e5247) #xc34177856ebca490))
(constraint (= (f #xeb602eeb78506106) #x0000eb602eeb7850))
(constraint (= (f #x64722ed10443d830) #x000064722ed10443))
(constraint (= (f #xd469450271dd7b4d) #xa8d28a04e3baf69c))
(constraint (= (f #x101666647464c484) #x0000101666647464))
(constraint (= (f #xe2e7948a2961693c) #x0000e2e7948a2961))
(constraint (= (f #x3380021995303d61) #x670004332a607ac4))
(constraint (= (f #x514e752eea84dec1) #xa29cea5dd509bd84))
(constraint (= (f #xc5ba9ac001ca04ac) #x0000c5ba9ac001ca))
(constraint (= (f #x3a278ec28a54aede) #x00003a278ec28a54))
(constraint (= (f #x9c3e36ac430bbded) #x387c6d5886177bdc))
(constraint (= (f #x596346c8d9783e27) #xb2c68d91b2f07c50))
(constraint (= (f #xba0b41a831e62416) #x0000ba0b41a831e6))
(constraint (= (f #xadc847e6bcc9e4e6) #x0000adc847e6bcc9))
(constraint (= (f #x709955e64c398e34) #x0000709955e64c39))
(constraint (= (f #xdd3eeee3a8e6c8cc) #x0000dd3eeee3a8e6))
(constraint (= (f #x46eeb4e2bd4433d2) #x000046eeb4e2bd44))
(constraint (= (f #xe20e129ec3dc2397) #xc41c253d87b84730))
(constraint (= (f #x2e9be5a76577e32c) #x00002e9be5a76577))
(constraint (= (f #x83eed4c1e9ecd2d6) #x000083eed4c1e9ec))
(constraint (= (f #xa700be30c887e249) #x4e017c61910fc494))
(constraint (= (f #xe75300e6ec14b523) #xcea601cdd8296a48))
(constraint (= (f #xeeb4ba89e6c71438) #x0000eeb4ba89e6c7))
(constraint (= (f #x8ed3318e497b0ca4) #x00008ed3318e497b))
(constraint (= (f #x18e29d7ebda29494) #x000018e29d7ebda2))
(constraint (= (f #xd1e1a03b88907651) #xa3c340771120eca4))
(constraint (= (f #x56ea32e015dc7770) #x000056ea32e015dc))
(constraint (= (f #x33698e4e24a777ea) #x000033698e4e24a7))
(constraint (= (f #xe8ced198b5e289ad) #xd19da3316bc5135c))
(constraint (= (f #x3170680a59744135) #x62e0d014b2e8826c))
(constraint (= (f #x74357c984b0e1b94) #x000074357c984b0e))
(constraint (= (f #x9d7b2c87d99c0b91) #x3af6590fb3381724))
(constraint (= (f #x4b1e9d526040196c) #x00004b1e9d526040))
(constraint (= (f #x80e0a3c9e42eede2) #x000080e0a3c9e42e))
(constraint (= (f #x61d28b43e5e6553c) #x000061d28b43e5e6))
(constraint (= (f #xcb816c248569d64b) #x9702d8490ad3ac98))
(constraint (= (f #x6b1e2eb823966462) #x00006b1e2eb82396))
(constraint (= (f #x5537dedd5bc9c7dd) #xaa6fbdbab7938fbc))
(constraint (= (f #x17830b71be3a57e3) #x2f0616e37c74afc8))
(constraint (= (f #x0e397aeee0b5dc05) #x1c72f5ddc16bb80c))
(constraint (= (f #xee06c4bd11392082) #x0000ee06c4bd1139))
(constraint (= (f #x65e59560869090de) #x000065e595608690))
(constraint (= (f #x4e06e691da0adb43) #x9c0dcd23b415b688))
(constraint (= (f #xe82b5c469883ce01) #xd056b88d31079c04))
(constraint (= (f #x122179b07b8d1d7a) #x0000122179b07b8d))
(constraint (= (f #x8d36b6c64b9474d2) #x00008d36b6c64b94))
(constraint (= (f #xce674d467ee1e9c8) #x0000ce674d467ee1))
(constraint (= (f #xe549e0e28a7a1cce) #x0000e549e0e28a7a))
(constraint (= (f #x751745adbcc9a38c) #x0000751745adbcc9))
(constraint (= (f #xc00c06e8ea707ee1) #x80180dd1d4e0fdc4))
(constraint (= (f #x644dba3a0ea945b1) #xc89b74741d528b64))
(constraint (= (f #xa81edc6a5ecb1c12) #x0000a81edc6a5ecb))
(constraint (= (f #x6979b08eeeae2d50) #x00006979b08eeeae))
(constraint (= (f #x9537eab81925ce65) #x2a6fd570324b9ccc))
(constraint (= (f #xe36c43e07636d3a6) #x0000e36c43e07636))
(constraint (= (f #xbacb93ec6e93ed9c) #x0000bacb93ec6e93))
(constraint (= (f #xcb2d9e279d3d1816) #x0000cb2d9e279d3d))
(constraint (= (f #x9d0b4ee9b33a4580) #x00009d0b4ee9b33a))
(constraint (= (f #x8c86de08b2e5e682) #x00008c86de08b2e5))
(constraint (= (f #x418c60eeebe723a7) #x8318c1ddd7ce4750))
(constraint (= (f #xd69044771ab0b924) #x0000d69044771ab0))
(constraint (= (f #x2ac22317aa049ed7) #x5584462f54093db0))
(constraint (= (f #xde8a489464b9657e) #x0000de8a489464b9))
(constraint (= (f #x159e409d2e38183a) #x0000159e409d2e38))
(constraint (= (f #x3134e95e4c11d040) #x00003134e95e4c11))
(constraint (= (f #xb4857a6e7e6357d3) #x690af4dcfcc6afa8))
(constraint (= (f #x190c20a3677230a9) #x32184146cee46154))
(constraint (= (f #xde2d13b7e798cc96) #x0000de2d13b7e798))
(constraint (= (f #x1dea4bae53e2e704) #x00001dea4bae53e2))
(constraint (= (f #x93dbaedecc54d290) #x000093dbaedecc54))
(constraint (= (f #x9d8d3b81aacc1be6) #x00009d8d3b81aacc))
(constraint (= (f #x4e26b2536eaa42a3) #x9c4d64a6dd548548))
(constraint (= (f #x9e792ca8e4d765b6) #x00009e792ca8e4d7))
(constraint (= (f #x0eb188ec017e507b) #x1d6311d802fca0f8))
(constraint (= (f #xe0e8a4303bb65873) #xc1d14860776cb0e8))

(check-synth)

