(set-logic BV)

(define-fun ehad ((x (BitVec 64))) (BitVec 64)
    (bvlshr x #x0000000000000001))
(define-fun arba ((x (BitVec 64))) (BitVec 64)
    (bvlshr x #x0000000000000004))
(define-fun shesh ((x (BitVec 64))) (BitVec 64)
    (bvlshr x #x0000000000000010))
(define-fun smol ((x (BitVec 64))) (BitVec 64)
    (bvshl x #x0000000000000001))
(define-fun im ((x (BitVec 64)) (y (BitVec 64)) (z (BitVec 64))) (BitVec 64)
    (ite (= x #x0000000000000001) y z))
(synth-fun f ((x (BitVec 64))) (BitVec 64)
    ((Start (BitVec 64) (#x0000000000000000 #x0000000000000001 x (bvnot Start) (smol Start) (ehad Start) (arba Start) (shesh Start) (bvand Start Start) (bvor Start Start) (bvxor Start Start) (bvadd Start Start) (im Start Start Start)))))

(constraint (= (f #x164e8d7d3a4ca9cc) #x000002c9d1afa749))
(constraint (= (f #x526d676564e0ca71) #x526d676564e0ca71))
(constraint (= (f #x13ebca3d3d6c424b) #x0000027d7947a7ad))
(constraint (= (f #x3535d02b9d5e8c9d) #x3535d02b9d5e8c9d))
(constraint (= (f #x9e57e0967e673d76) #x9e57e0967e673d77))
(constraint (= (f #x33bee807b4e5e105) #x33bee807b4e5e105))
(constraint (= (f #xeb7eb7ed0e0084c8) #x00001d6fd6fda1c0))
(constraint (= (f #x0596da962c5736c4) #x000000b2db52c58a))
(constraint (= (f #x757e8e2d23b655d4) #x00000eafd1c5a476))
(constraint (= (f #x7976e9e747ee2062) #x7976e9e747ee2063))
(constraint (= (f #x3499aee4d184e41e) #x3499aee4d184e41f))
(constraint (= (f #x25e807a83d1c17d4) #x000004bd00f507a3))
(constraint (= (f #xd0abda23ece3d168) #x00001a157b447d9c))
(constraint (= (f #xd8a9a6d156e57b54) #x00001b1534da2adc))
(constraint (= (f #x611e514aac9dee13) #x00000c23ca295593))
(constraint (= (f #x3e7d577beea84159) #x3e7d577beea84159))
(constraint (= (f #x78170e7116e97251) #x78170e7116e97251))
(constraint (= (f #x7e89843c068e1dbe) #x7e89843c068e1dbf))
(constraint (= (f #x9ae6e91deeeaaa94) #x0000135cdd23bddd))
(constraint (= (f #x69e174891e0eed17) #x00000d3c2e9123c1))
(constraint (= (f #x7e6892a4ce1d5754) #x00000fcd125499c3))
(constraint (= (f #x0de5ae6412074513) #x000001bcb5cc8240))
(constraint (= (f #x78950ec6b90ee622) #x78950ec6b90ee623))
(constraint (= (f #xa7703eec4ed4e3a7) #x000014ee07dd89da))
(constraint (= (f #x9e6218bc6ae2ab38) #x000013cc43178d5c))
(constraint (= (f #x52c8057d7123a91b) #x00000a5900afae24))
(constraint (= (f #x345eedacb62ae27c) #x0000068bddb596c5))
(constraint (= (f #xd966098d4a102d99) #xd966098d4a102d99))
(constraint (= (f #x001be56c93d71dc8) #x000000037cad927a))
(constraint (= (f #x69ced1b037e528ce) #x69ced1b037e528cf))
(constraint (= (f #xaac7ec91e339bd72) #xaac7ec91e339bd73))
(constraint (= (f #xdecdd19e7ee59c9a) #xdecdd19e7ee59c9b))
(constraint (= (f #x52d28e14eab36612) #x52d28e14eab36613))
(constraint (= (f #xdc719eb64333dc1a) #xdc719eb64333dc1b))
(constraint (= (f #x569551d1a3d6a4eb) #x00000ad2aa3a347a))
(constraint (= (f #xe4c905162e11e95a) #xe4c905162e11e95b))
(constraint (= (f #xe0e49ade8ad3a676) #xe0e49ade8ad3a677))
(constraint (= (f #xc294d8c4148543e5) #xc294d8c4148543e5))
(constraint (= (f #xed75e5de426e07d4) #x00001daebcbbc84d))
(constraint (= (f #xe24ceb882cebb759) #xe24ceb882cebb759))
(constraint (= (f #xad8dd1a3b90717ab) #x000015b1ba347720))
(constraint (= (f #x3e64ecd28e11422d) #x3e64ecd28e11422d))
(constraint (= (f #x95d1537e813ad9eb) #x000012ba2a6fd027))
(constraint (= (f #xce27357ee2ea0e34) #x000019c4e6afdc5d))
(constraint (= (f #x2aee47c87e1120b6) #x2aee47c87e1120b7))
(constraint (= (f #x0ddbcb493be205a3) #x000001bb7969277c))
(constraint (= (f #x41852c75e8e50cd2) #x41852c75e8e50cd3))
(constraint (= (f #x37cbe5d0c0b756ee) #x37cbe5d0c0b756ef))
(constraint (= (f #xe8741438462a3d47) #x00001d0e828708c5))
(constraint (= (f #xb64eb73326462246) #xb64eb73326462247))
(constraint (= (f #x90be22ee17a49ede) #x90be22ee17a49edf))
(constraint (= (f #x741c308b4d059064) #x00000e83861169a0))
(constraint (= (f #xca0eb6cbe124c325) #xca0eb6cbe124c325))
(constraint (= (f #x9e1625401821e09e) #x9e1625401821e09f))
(constraint (= (f #x7825ee9e5665a5b6) #x7825ee9e5665a5b7))
(constraint (= (f #xae79300375d73e23) #x000015cf26006eba))
(constraint (= (f #xaeaa1e4962419934) #x000015d543c92c48))
(constraint (= (f #x0c3dee621b17b2b4) #x00000187bdcc4362))
(constraint (= (f #xd8433ea7a59e4626) #xd8433ea7a59e4627))
(constraint (= (f #x7653e9cc77e77de2) #x7653e9cc77e77de3))
(constraint (= (f #x2491dc327915d077) #x000004923b864f22))
(constraint (= (f #xc684039d49d62a0a) #xc684039d49d62a0b))
(constraint (= (f #x0b366dbd2d65a38a) #x0b366dbd2d65a38b))
(constraint (= (f #x36bcde43862d42ab) #x000006d79bc870c5))
(constraint (= (f #x30a51e89e9b5d417) #x00000614a3d13d36))
(constraint (= (f #x91ab9e8244593193) #x0000123573d0488b))
(constraint (= (f #x246e95db89091413) #x0000048dd2bb7121))
(constraint (= (f #x06b06aee99156ec6) #x06b06aee99156ec7))
(constraint (= (f #x04a0c586ecbd39e8) #x0000009418b0dd97))
(constraint (= (f #xeba9ed7ec47125e4) #x00001d753dafd88e))
(constraint (= (f #x1d32c1eeaa11d992) #x1d32c1eeaa11d993))
(constraint (= (f #xeae1d06c766b70ae) #xeae1d06c766b70af))
(constraint (= (f #xca8c0de00310ea28) #x0000195181bc0062))
(constraint (= (f #xa0ec7a8b259be2de) #xa0ec7a8b259be2df))
(constraint (= (f #xe02eb5801ee986a5) #xe02eb5801ee986a5))
(constraint (= (f #x43acae0de645503a) #x43acae0de645503b))
(constraint (= (f #x1793305ea50e03d9) #x1793305ea50e03d9))
(constraint (= (f #x86455b9135095931) #x86455b9135095931))
(constraint (= (f #x0daeb70d6024a2ad) #x0daeb70d6024a2ad))
(constraint (= (f #x7c2aebb9a821715b) #x00000f855d773504))
(constraint (= (f #xa74cec689ba7d207) #x000014e99d8d1374))
(constraint (= (f #x3178c8ee64bacdd8) #x0000062f191dcc97))
(constraint (= (f #xb28e96d64a3b0e30) #x00001651d2dac947))
(constraint (= (f #x578980bd0e5996eb) #x00000af13017a1cb))
(constraint (= (f #xe55aae47588a2cd3) #x00001cab55c8eb11))
(constraint (= (f #xc25b388a3eedabeb) #x0000184b671147dd))
(constraint (= (f #xca58490ac46365d4) #x0000194b0921588c))
(constraint (= (f #xa8e5ea826242a0ce) #xa8e5ea826242a0cf))
(constraint (= (f #x90ee1e11889c3710) #x0000121dc3c23113))
(constraint (= (f #x9e4e96d096a3a6ad) #x9e4e96d096a3a6ad))
(constraint (= (f #x4d15b74058060684) #x000009a2b6e80b00))
(constraint (= (f #x7533dec9de443361) #x7533dec9de443361))
(constraint (= (f #xeec434eeee2eb832) #xeec434eeee2eb833))
(constraint (= (f #xee67148d0aeceb41) #xee67148d0aeceb41))
(constraint (= (f #xc1cb2ce7c8ee850b) #x00001839659cf91d))
(constraint (= (f #x775a5ee0a6e37c64) #x00000eeb4bdc14dc))
(constraint (= (f #x019032802e7b2343) #x00000032065005cf))
(constraint (= (f #x101e62c10eb78723) #x00000203cc5821d6))
(constraint (= (f #x71d1ed5e3ce6ba61) #x71d1ed5e3ce6ba61))
(constraint (= (f #x9beddd429451e57c) #x0000137dbba8528a))

(check-synth)

