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

(constraint (= (f #x10892BD4102D7B82) #x004224AF5040B5EE))
(constraint (= (f #x05ECF3F0030A0B42) #x0017B3CFC00C282D))
(constraint (= (f #xEEBC361301D919C7) #x03BAF0D84C076467))
(constraint (= (f #xF1486B08608F5706) #x03C521AC21823D5C))
(constraint (= (f #x022C595F4F6B7C4D) #x0008B1657D3DADF1))
(constraint (= (f #x9ACDC06DF989CEF0) #x0AF5640B60A9A531))
(constraint (= (f #xB0BA744E79599031) #x0D1CE9CD28BEAB05))
(constraint (= (f #xC40FFDFA260E58F7) #x04C10060E6A12E91))
(constraint (= (f #xFDFD72014C860E75) #x006079603D58A129))
(constraint (= (f #x6229F54037F62879) #x0A67A1FC0581A788))
(constraint (= (f #x0000000000135F8F) #x0000000000000000))
(constraint (= (f #x000000000018A34D) #x0000000000000000))
(constraint (= (f #x00000000001B1E0D) #x0000000000000000))
(constraint (= (f #x0000000000174D0A) #x0000000000000000))
(constraint (= (f #x00000000001570C4) #x0000000000000000))
(constraint (= (f #x0000000000002F40) #x00000000000000BD))
(constraint (= (f #x000000000000BE4C) #x00000000000002F9))
(constraint (= (f #x000000000000BF00) #x00000000000002FC))
(constraint (= (f #x0000000000005081) #x0000000000000142))
(constraint (= (f #x00000000000024C2) #x0000000000000093))
(constraint (= (f #x9D722B380F63DEEC) #x0275C8ACE03D8F7B))
(constraint (= (f #x1BB0AFE175768DAA) #x006EC2BF85D5DA36))
(constraint (= (f #x2DFEA0194227156A) #x00B7FA8065089C55))
(constraint (= (f #x018E3EB379F16025) #x000638FACDE7C580))
(constraint (= (f #xADC5644FC60173AB) #x02B715913F1805CE))
(constraint (= (f #x000000000015E3BC) #x000000000003E24C))
(constraint (= (f #x0000000000190530) #x000000000002B0F5))
(constraint (= (f #x00000000001D67F9) #x0000000000027A80))
(constraint (= (f #x000000000012A7F2) #x0000000000037E81))
(constraint (= (f #x00000000001666B1) #x000000000003AABD))
(constraint (= (f #x0000000000006A7F) #x0000000000001550))
(constraint (= (f #x0000000000007B74) #x0000000000001092))
(constraint (= (f #x000000000000E5B4) #x000000000000254A))
(constraint (= (f #x0000000000003379) #x0000000000000A91))
(constraint (= (f #x000000000000D934) #x000000000000252A))
(constraint (= (f #x16B5D146FF75E39D) #x03BDE73CB019E24A))
(constraint (= (f #x2B54D4C47ECD7C97) #x07DFD7D4C835785B))
(constraint (= (f #x547DAE2A3061F7DE) #x0FC86F27E50A2186))
(constraint (= (f #xE0E4F169DAFB8E54) #x0212D13BA6F0C92F))
(constraint (= (f #x72533E187575BAD5) #x096F542289F9ECF7))
(constraint (= (f #x0000000000161D6A) #x0000000000000000))
(constraint (= (f #x00000000001EEB6D) #x0000000000000000))
(constraint (= (f #x00000000001EE9A2) #x0000000000000000))
(constraint (= (f #x0000000000165167) #x0000000000000000))
(constraint (= (f #x00000000001EC02D) #x0000000000000000))
(constraint (= (f #x0000000000002E2D) #x00000000000000B8))
(constraint (= (f #x00000000000038A4) #x00000000000000E2))
(constraint (= (f #x0000000000006E2D) #x00000000000001B8))
(constraint (= (f #x000000000000A065) #x0000000000000281))
(constraint (= (f #x0000000000000421) #x0000000000000010))
(constraint (= (f #x000000000018A2D8) #x0000000000029E76))
(constraint (= (f #x0000000000145115) #x000000000003CF33))
(constraint (= (f #x000000000018B290) #x0000000000029D7B))
(constraint (= (f #x0000000000184293) #x0000000000028C7B))
(constraint (= (f #x000000000014EE1A) #x000000000003D322))
(constraint (= (f #x000000000000D997) #x0000000000002555))
(constraint (= (f #x000000000000BD53) #x00000000000028AA))
(constraint (= (f #x000000000000FE19) #x0000000000002045))
(constraint (= (f #x000000000000315F) #x0000000000000A54))
(constraint (= (f #x0000000000004992) #x0000000000000952))
(constraint (= (f #xF250F5AB26126739) #x016F11EFD6A36A94))
(constraint (= (f #xDF421F0F2329AD26) #x037D087C3C8CA6B4))
(constraint (= (f #x1E2BF9455DCB9470) #x0227C0BCFE65CBC9))
(constraint (= (f #x429DB058684C20BC) #x0C7A6D0E8B8D461C))
(constraint (= (f #xB6E73C0ABDA2C2AB) #x02DB9CF02AF68B0A))
(constraint (= (f #x6BB86D07DF1E3150) #x0BCC8B708612253F))
(constraint (= (f #x14DC5AF6C2E8C55B) #x03D64EF1B47394FE))
(constraint (= (f #x610CA8C3ECAB7DFA) #x0A315F94435FD860))
(constraint (= (f #x16FA53D3BE0BAA22) #x005BE94F4EF82EA8))
(constraint (= (f #x017EE661B13945BC) #x003832AA2D34BCEC))
(constraint (= (f #x000000000000D230) #x0000000000002A4A))
(constraint (= (f #x00000000001D432F) #x0000000000000000))
(constraint (= (f #x00000000001E4EF7) #x0000000000022D31))
(constraint (= (f #x0000000000008143) #x0000000000000205))
(constraint (= (f #x000000000000F73B) #x0000000000002128))
(constraint (= (f #x0000000000008E4A) #x0000000000000239))
(constraint (= (f #x000000000018CCD9) #x0000000000029556))
(constraint (= (f #x00000000001FB2C2) #x0000000000000000))
(constraint (= (f #x00000000001C1C32) #x0000000000024245))
(constraint (= (f #x000000000000A69F) #x0000000000001554))
(constraint (= (f #x000000000000E361) #x000000000000038D))
(constraint (= (f #x000000000000042E) #x0000000000000010))
(constraint (= (f #x0F6762D26B8D580F) #x003D9D8B49AE3560))
(constraint (= (f #x0000000000010DF1) #x0000000000003161))
(constraint (= (f #x0000000000005679) #x0000000000001551))
(constraint (= (f #x8000000000007374) #x0800000000000959))
(constraint (= (f #x000000000000153B) #x00000000000002A8))
(constraint (= (f #x4000000000007595) #x0C000000000009EB))
(constraint (= (f #x0000000000014893) #x0000000000003D9B))
(constraint (= (f #x67F979A53C30FDDE) #x0A80B8AEF4451066))
(constraint (= (f #x800000000000DA75) #x08000000000016E9))
(constraint (= (f #xC000000000006B7B) #x0400000000000BD8))
(constraint (= (f #x0000000000005D97) #x0000000000001455))
(constraint (= (f #x0000000000005F91) #x0000000000001412))

(check-synth)

