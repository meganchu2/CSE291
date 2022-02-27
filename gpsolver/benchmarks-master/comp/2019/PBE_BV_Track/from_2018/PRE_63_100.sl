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

(constraint (= (f #x15e6c2e77193abba) #xea193d188e6c5444))
(constraint (= (f #x3d297e945807958b) #xc2d6816ba7f86a75))
(constraint (= (f #x6eebcaea31c19eba) #x91143515ce3e6144))
(constraint (= (f #xcdcdebcc40612a71) #x32321433bf9ed58f))
(constraint (= (f #x6eacdde2d1cb55da) #x9153221d2e34aa24))
(constraint (= (f #x56bc8684db94ce51) #x56bc8684db94ce50))
(constraint (= (f #x09c26e4c62d94e6e) #xf63d91b39d26b190))
(constraint (= (f #xb4ac67c4370a8304) #xb4ac67c4370a8304))
(constraint (= (f #x53eee6d39dd1e47e) #xac11192c622e1b80))
(constraint (= (f #xc112c3be11728b4d) #xc112c3be11728b4c))
(constraint (= (f #xc24ec2432c1130ae) #x3db13dbcd3eecf50))
(constraint (= (f #x1480de45ca4bb419) #xeb7f21ba35b44be7))
(constraint (= (f #x277ce5cd9b51d2ea) #xd8831a3264ae2d14))
(constraint (= (f #xdd67a6599e951e88) #x229859a6616ae176))
(constraint (= (f #x56eea2d0eae88e49) #x56eea2d0eae88e48))
(constraint (= (f #xc6c4ba7526111de5) #x393b458ad9eee21b))
(constraint (= (f #x5aee9ee245e9cba6) #xa511611dba163458))
(constraint (= (f #x13d0ca0035770b08) #xec2f35ffca88f4f6))
(constraint (= (f #x10e61ecc812369a4) #xef19e1337edc965a))
(constraint (= (f #xeee8dab47e7b7888) #x1117254b81848776))
(constraint (= (f #x8c2ee378e7b7ece2) #x73d11c871848131c))
(constraint (= (f #x22e285921c1ca2a0) #x22e285921c1ca2a0))
(constraint (= (f #x56c701ed03e517c9) #xa938fe12fc1ae837))
(constraint (= (f #xbb16cb3e3d266ad9) #xbb16cb3e3d266ad8))
(constraint (= (f #x51d2e2c0ebdd12ae) #xae2d1d3f1422ed50))
(constraint (= (f #xd6b653aae0569748) #xd6b653aae0569748))
(constraint (= (f #x560e031e6c09e620) #xa9f1fce193f619de))
(constraint (= (f #xe5d7e6e7086205eb) #xe5d7e6e7086205ea))
(constraint (= (f #xa9d62ae4ee2decea) #x5629d51b11d21314))
(constraint (= (f #x3ec561e4ece42545) #x3ec561e4ece42544))
(constraint (= (f #x483e78aed53109ac) #xb7c187512acef652))
(constraint (= (f #x6a34629e1ea4c49c) #x6a34629e1ea4c49c))
(constraint (= (f #xe2503bc028d94160) #x1dafc43fd726be9e))
(constraint (= (f #x27766863a303965e) #xd889979c5cfc69a0))
(constraint (= (f #x591e9b614e2716e3) #xa6e1649eb1d8e91d))
(constraint (= (f #x35429a18d1203821) #x35429a18d1203820))
(constraint (= (f #x22462e2e0ab29ed9) #x22462e2e0ab29ed8))
(constraint (= (f #xda736ceeca2d5925) #x258c931135d2a6db))
(constraint (= (f #x112e99cb554e5211) #x112e99cb554e5210))
(constraint (= (f #x30975ca46253bbd7) #xcf68a35b9dac4429))
(constraint (= (f #x9cd47886ee32e024) #x9cd47886ee32e024))
(constraint (= (f #xd386a7aba5256de2) #x2c7958545ada921c))
(constraint (= (f #xe534783c6e08e1ec) #xe534783c6e08e1ec))
(constraint (= (f #x52c417e6e968b947) #x52c417e6e968b946))
(constraint (= (f #x4de50b6c811e2dcd) #x4de50b6c811e2dcc))
(constraint (= (f #xe9c4de6052c9e19e) #x163b219fad361e60))
(constraint (= (f #x899019851839e034) #x766fe67ae7c61fca))
(constraint (= (f #xcdee1531be2cec2a) #xcdee1531be2cec2a))
(constraint (= (f #xd4b7e76ec3da2282) #xd4b7e76ec3da2282))
(constraint (= (f #xe5d2908234a7d2b6) #x1a2d6f7dcb582d48))
(constraint (= (f #xe7e9ee010455235b) #x181611fefbaadca5))
(constraint (= (f #x7072aea236d8723e) #x7072aea236d8723e))
(constraint (= (f #x67cb5e7c11766c1e) #x67cb5e7c11766c1e))
(constraint (= (f #x1b533e9e646dd70e) #xe4acc1619b9228f0))
(constraint (= (f #x3a65c0a76b5eeca1) #x3a65c0a76b5eeca0))
(constraint (= (f #x4b8b29ac603ed4d2) #x4b8b29ac603ed4d2))
(constraint (= (f #x9ca252c09836c743) #x9ca252c09836c742))
(constraint (= (f #x66951ceb568a10cc) #x66951ceb568a10cc))
(constraint (= (f #x12c7c0c51eeec9d5) #x12c7c0c51eeec9d4))
(constraint (= (f #x336454e8ea57dea8) #xcc9bab1715a82156))
(constraint (= (f #xba9c8a2d0caaa6ce) #xba9c8a2d0caaa6ce))
(constraint (= (f #x886ee1e171e78a72) #x77911e1e8e18758c))
(constraint (= (f #xad54c4142ea749b3) #x52ab3bebd158b64d))
(constraint (= (f #x6be8dcee01da4e54) #x6be8dcee01da4e54))
(constraint (= (f #x3e2a3128b9781e24) #x3e2a3128b9781e24))
(constraint (= (f #x438746e28e8d41b6) #xbc78b91d7172be48))
(constraint (= (f #x1363cae3aee9c475) #xec9c351c51163b8b))
(constraint (= (f #x75a8885eca358e1d) #x8a5777a135ca71e3))
(constraint (= (f #xa56ea5b34b1277e3) #xa56ea5b34b1277e2))
(constraint (= (f #x72c69973520aee67) #x72c69973520aee66))
(constraint (= (f #x24ce391db816eeaa) #x24ce391db816eeaa))
(constraint (= (f #x2be3c1e13515044e) #xd41c3e1ecaeafbb0))
(constraint (= (f #x4a9eee55e6a812ee) #x4a9eee55e6a812ee))
(constraint (= (f #x2b205924572ea213) #x2b205924572ea212))
(constraint (= (f #x5e572c5705409e8d) #x5e572c5705409e8c))
(constraint (= (f #x4cd151bda26d3791) #xb32eae425d92c86f))
(constraint (= (f #x5092c01e9ea670ed) #x5092c01e9ea670ec))
(constraint (= (f #x6b1948a7c2da0315) #x6b1948a7c2da0314))
(constraint (= (f #x462e8407cad38860) #xb9d17bf8352c779e))
(constraint (= (f #x6c5dcbede763cc89) #x93a23412189c3377))
(constraint (= (f #x6d149ba6bec82157) #x6d149ba6bec82156))
(constraint (= (f #xd0e239675e657988) #x2f1dc698a19a8676))
(constraint (= (f #xa7646a29cce5e342) #x589b95d6331a1cbc))
(constraint (= (f #xd81845c48225de1b) #x27e7ba3b7dda21e5))
(constraint (= (f #xc2213444d43ba10d) #x3ddecbbb2bc45ef3))
(constraint (= (f #x8042bc8d601e53c8) #x8042bc8d601e53c8))
(constraint (= (f #xb71ac9eeb0de028e) #xb71ac9eeb0de028e))
(constraint (= (f #x809bc7cae9e1561e) #x7f643835161ea9e0))
(constraint (= (f #x4c944b35ae826e1e) #x4c944b35ae826e1e))
(constraint (= (f #xeab351516ba85e53) #xeab351516ba85e52))
(constraint (= (f #xdecbe821ed58ae25) #xdecbe821ed58ae24))
(constraint (= (f #x9465837753de0c8b) #x9465837753de0c8a))
(constraint (= (f #x26e582e370c9e1a0) #xd91a7d1c8f361e5e))
(constraint (= (f #xde62c7127e117683) #x219d38ed81ee897d))
(constraint (= (f #x9193a592e29dc1ea) #x6e6c5a6d1d623e14))
(constraint (= (f #x455d61cb8eec5701) #x455d61cb8eec5700))
(constraint (= (f #x7ca6301b1ebcbe0e) #x7ca6301b1ebcbe0e))
(constraint (= (f #x3793c0392b3be135) #xc86c3fc6d4c41ecb))
(constraint (= (f #xe8131d67dbda48a3) #xe8131d67dbda48a2))
(constraint (= (f #xe2b53179744c91e2) #xe2b53179744c91e2))

(check-synth)

