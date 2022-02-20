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

(constraint (= (f #x9b8e2e90a92dda4b) #x371c5d21525bb497))
(constraint (= (f #x6847ad7318bc18b6) #xd08f5ae63178316d))
(constraint (= (f #xcc41853e2892b25e) #x98830a7c512564bd))
(constraint (= (f #x648226881abee165) #xc9044d10357dc2cb))
(constraint (= (f #xe8ab524a993a81ac) #xd156a49532750359))
(constraint (= (f #x034825baa2a1eea4) #xfc8357e9b333f271))
(constraint (= (f #x6ebd07aee93268ae) #xdd7a0f5dd264d15d))
(constraint (= (f #xcaa0c366e795eba6) #x28b53062a9f0b59f))
(constraint (= (f #x82c1111674de9e89) #x0582222ce9bd3d13))
(constraint (= (f #x42512e1c8855c2a1) #x84a25c3910ab8543))
(constraint (= (f #xc05b52b7595b5e15) #x80b6a56eb2b6bc2b))
(constraint (= (f #x68e3dca80b24a49a) #xd1c7b95016494935))
(constraint (= (f #x249828530e7d039d) #x493050a61cfa073b))
(constraint (= (f #x8b9b5533637c3be8) #x1736aa66c6f877d1))
(constraint (= (f #x5ed4be8b31442d38) #xbda97d1662885a71))
(constraint (= (f #x114deb6de25cb67e) #x229bd6dbc4b96cfd))
(constraint (= (f #xecc5247a98adc0cc) #x046e893dbdc76327))
(constraint (= (f #x3445639567b3bb18) #xc876463141d10936))
(constraint (= (f #x38112aea45790e80) #xc46dc267162f6097))
(constraint (= (f #x4ee7e2a4c110ed01) #x9dcfc5498221da03))
(constraint (= (f #x23b29979e49bba25) #x476532f3c937744b))
(constraint (= (f #x59442a1cec6b6b1d) #xb2885439d8d6d63b))
(constraint (= (f #x8d3be1a562e6a5ac) #x1a77c34ac5cd4b59))
(constraint (= (f #xb72adee5ad2e1ead) #x6e55bdcb5a5c3d5b))
(constraint (= (f #x8650b284169dc202) #x714a4253a7f861dd))
(constraint (= (f #xb919ec4ceacee770) #x7233d899d59dcee1))
(constraint (= (f #x78abc6d1c6e16e06) #x7fc97cc11cb07b19))
(constraint (= (f #x9990716ed056073c) #x3320e2dda0ac0e79))
(constraint (= (f #x5a5ee96b138e8b26) #xb4bdd2d6271d164d))
(constraint (= (f #x3a4d3b7a4e5d2c75) #x749a76f49cba58eb))
(constraint (= (f #x0b725cac4d0c69e9) #x16e4b9589a18d3d3))
(constraint (= (f #xb1504768455e6ad5) #x62a08ed08abcd5ab))
(constraint (= (f #x08b3e4b6eae3a646) #xf6c0dcfda66e1f55))
(constraint (= (f #x1ee3197753e53bbc) #xdf2eb4f136dc7088))
(constraint (= (f #xdeea4b883be23e55) #xbdd4971077c47cab))
(constraint (= (f #x4ee689447bd83e8a) #x9dcd1288f7b07d15))
(constraint (= (f #xc21a5867e6518571) #x8434b0cfcca30ae3))
(constraint (= (f #x5961aecea0e21ae8) #xb2c35d9d41c435d1))
(constraint (= (f #xbb92cc23d33ea373) #x77259847a67d46e7))
(constraint (= (f #x8cbe6a937eb0703e) #x197cd526fd60e07d))
(constraint (= (f #x16b34619d7d1eec5) #x2d668c33afa3dd8b))
(constraint (= (f #x473858c5e52310ec) #xb45421adbc8abe05))
(constraint (= (f #x8e3e507d01670a2a) #x68ddca7b2e828533))
(constraint (= (f #xa7d9e96abe76babc) #x4fb3d2d57ced7579))
(constraint (= (f #xb046793a2b8ea0da) #x608cf274571d41b5))
(constraint (= (f #x1dce1e1041ec01b0) #x3b9c3c2083d80361))
(constraint (= (f #xed130207dc418eed) #xda26040fb8831ddb))
(constraint (= (f #x7188d2244ec39531) #xe311a4489d872a63))
(constraint (= (f #xda8a813e0be7ce6e) #x17ccd6ae1359b4ab))
(constraint (= (f #xe7ecec2771e04eb4) #xcfd9d84ee3c09d69))
(constraint (= (f #x665319c57835d896) #x9347b49e3046c9e0))
(constraint (= (f #xe48a4ea60d763e16) #xc9149d4c1aec7c2d))
(constraint (= (f #x2e1145729928393d) #x5c228ae53250727b))
(constraint (= (f #x77d9e3e0d1e2c02d) #xefb3c7c1a3c5805b))
(constraint (= (f #x7654b73ba7be0e9b) #xeca96e774f7c1d37))
(constraint (= (f #xedd1e90225c09246) #xdba3d2044b81248d))
(constraint (= (f #xb2c9a3b60e1e85ea) #x6593476c1c3d0bd5))
(constraint (= (f #x205767010be585a6) #xdda3228ee35c21ff))
(constraint (= (f #x9e11c805e24562c6) #x580d1b79bf96470d))
(constraint (= (f #xe1d6a8be5a474505) #xc3ad517cb48e8a0b))
(constraint (= (f #x16be5a25ce199dad) #x2d7cb44b9c333b5b))
(constraint (= (f #xdbeab3114506930e) #xb7d566228a0d261d))
(constraint (= (f #x8c99e16e4eb7b163) #x1933c2dc9d6f62c7))
(constraint (= (f #x59b61111c31bbc67) #xb36c2223863778cf))
(constraint (= (f #x10289d85d18ed22d) #x20513b0ba31da45b))
(constraint (= (f #x46a19ca8a2e87d13) #x8d43395145d0fa27))
(constraint (= (f #x2b31ada268a7ebda) #xd21b378370cd9568))
(constraint (= (f #x7e159e4433e151b9) #xfc2b3c8867c2a373))
(constraint (= (f #xee43266d7e7c391d) #xdc864cdafcf8723b))
(constraint (= (f #x11adad4d8a748624) #x235b5a9b14e90c49))
(constraint (= (f #xbd8211431c2331cc) #x36a5cda8b21a9b17))
(constraint (= (f #xbcca2999bb9ea4e4) #x79945333773d49c9))
(constraint (= (f #x86436e152990007a) #x0c86dc2a532000f5))
(constraint (= (f #xe139b8e642799de5) #xc27371cc84f33bcb))
(constraint (= (f #x4b2222bed1deae1b) #x9644457da3bd5c37))
(constraint (= (f #xd5917ba8e52a1079) #xab22f751ca5420f3))
(constraint (= (f #x136e1bba33679dd2) #xeb5b028a2961e850))
(constraint (= (f #x9b2eec92094118e9) #x365dd924128231d3))
(constraint (= (f #x87a3b433dd63b1b7) #x0f476867bac7636f))
(constraint (= (f #x3a98ece9de187883) #x7531d9d3bc30f107))
(constraint (= (f #xe17dc46eede541ae) #x106a5f4a233c6a37))
(constraint (= (f #xa9b7d8ae1273ead2) #x4baca9c70c64d680))
(constraint (= (f #x3c75554e4a4ec14b) #x78eaaa9c949d8297))
(constraint (= (f #xe85eb54887cbd5a0) #x091b5f62efb76d05))
(constraint (= (f #x77a7d4a689882338) #xef4fa94d13104671))
(constraint (= (f #x343ee8312119578c) #xc87d294bccd512fb))
(constraint (= (f #x87b1b951780d9eec) #x6fd32b1970718725))
(constraint (= (f #x8a2aede591420519) #x1455dbcb22840a33))
(constraint (= (f #xca39d2931978996d) #x9473a52632f132db))
(constraint (= (f #x63ed85653e10c7c8) #xc7db0aca7c218f91))
(constraint (= (f #x85eda9483d78e271) #x0bdb52907af1c4e3))
(constraint (= (f #x4e40b8942a6cb745) #x9c81712854d96e8b))
(constraint (= (f #x823932e9a30074bd) #x047265d34600e97b))
(constraint (= (f #x3deed57a12ee8e26) #x7bddaaf425dd1c4d))
(constraint (= (f #xebbb420e97985ebd) #xd776841d2f30bd7b))
(constraint (= (f #x375129a175ebb875) #x6ea25342ebd770eb))
(constraint (= (f #x1921d64e99001c6b) #x3243ac9d320038d7))
(constraint (= (f #x2b02d5a4b08e94cd) #x5605ab49611d299b))
(constraint (= (f #x9d32970c7d38a299) #x3a652e18fa714533))
(constraint (= (f #x3c5ed807a7644ede) #x78bdb00f4ec89dbd))

(check-synth)

