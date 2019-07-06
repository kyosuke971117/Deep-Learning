import pandas as pd
import numpy as np

def kdd_load():
# ##################################
# # 学習用のデータをKDDTrain+から取得
# ##################################
    kdd_dataset = pd.read_csv('../../NSL-KDD_AAE_hara/NSL-KDD_AAE_hara/NSL_KDD_data/KDDTrain+.csv')

    #使用したネットワークサービスを数値に変換
    kdd_dataset['service']= kdd_dataset['service'].map({
    "ftp_data":0,
    "other":1,
    "private":2,
    "http":3,
    "remote_job":4,
    "name":5,
    "netbios_ns":6,
    "eco_i":7,
    "mtp":8,
    "telnet":9,
    "finger":10,
    "domain_u":11,
    "supdup":12,
    "uucp_path":13,
    "Z39_50":14,
    "smtp":15,
    "csnet_ns":16,
    "uucp":17,
    "netbios_dgm":18,
    "urp_i":19,
    "auth":20,
    "domain":21,
    "ftp":22,
    "bgp":23,
    "ldap":24,
    "ecr_i":25,
    "gopher":26,
    "vmnet":27,
    "systat":28,
    "http_443":29,
    "efs":30,
    "whois":31,
    "imap4":32,
    "iso_tsap":33,
    "echo":34,
    "klogin":35,
    "link":36,
    "sunrpc":37,
    "login":38,
    "kshell":39,
    "sql_net":40,
    "time":41,
    "hostnames":42,
    "exec":43,
    "ntp_u":44,
    "discard":45,
    "nntp":46,
    "courier":47,
    "ctf":48,
    "ssh":49,
    "daytime":50,
    "shell":51,
    "netstat":52,
    "pop_3":53,
    "nnsp":54,
    "IRC":55,
    "pop_2":56,
    "printer":57,
    "tim_i":58,
    "pm_dump":59,
    "red_i":60,
    "netbios_ssn":61,
    "rje":62,
    "X11":63,
    "urh_i":64,
    "http_8001":65,
    "aol":66,
    "http_2784":67,
    "tftp_u":68,
    "harvest":69
    })

    #その接続に使用したプロトコルを数値に変換
    kdd_dataset['protocol_type'] = kdd_dataset['protocol_type'].map({'tcp':0,'udp':1,'icmp':2})
    #その接続の状態を数値に変換
    kdd_dataset['flag'] = kdd_dataset['flag'].map({'SF':0,'S1':1,'REJ':2,'S2':3,'S0':4,'S3':5,'RSTO':6,'RSTR':7,'RSTOS0':8,'OTH':9,'SH':10})

    #攻撃を数値に置換
    kdd_dataset['AttackTypes'] = kdd_dataset['AttackTypes'].map({
    "normal":0,
    "back":1,
    "land":1,
    "neptune":1,
    "smurf":1,
    "pod":1,
    "teardrop":1,
    "buffer_overflow":1,
    "loadmodule":1,
    "perl":1,
    "rootkit":1,
    "ftp_write":1,
    "guess_passwd":1,
    "imap":1,
    "multihop":1,
    "phf":1,
    "spy":1,
    "warezclient":1,
    "warezmaster":1,
    "satan":1,
    "ipsweep":1,
    "portsweep":1,
    "nmap": 1
    })

    #ラベルごとの枚数をカウントする
    print('TrainDataCount-------------------------------------------------------------')
    num_eachAttack = kdd_dataset["AttackTypes"].value_counts() #要素の出現回数を格納
    print(num_eachAttack)

# ##################################
# # 学習用のデータをKDDTest+から取得
# ##################################
    kdd_dataset_test = pd.read_csv('../../NSL-KDD_AAE_hara/NSL-KDD_AAE_hara/NSL_KDD_data/KDDTest+.csv')

    kdd_dataset_test['service']= kdd_dataset_test['service'].map({
    "ftp_data":0,
    "other":1,
    "private":2,
    "http":3,
    "remote_job":4,
    "name":5,
    "netbios_ns":6,
    "eco_i":7,
    "mtp":8,
    "telnet":9,
    "finger":10,
    "domain_u":11,
    "supdup":12,
    "uucp_path":13,
    "Z39_50":14,
    "smtp":15,
    "csnet_ns":16,
    "uucp":17,
    "netbios_dgm":18,
    "urp_i":19,
    "auth":20,
    "domain":21,
    "ftp":22,
    "bgp":23,
    "ldap":24,
    "ecr_i":25,
    "gopher":26,
    "vmnet":27,
    "systat":28,
    "http_443":29,
    "efs":30,
    "whois":31,
    "imap4":32,
    "iso_tsap":33,
    "echo":34,
    "klogin":35,
    "link":36,
    "sunrpc":37,
    "login":38,
    "kshell":39,
    "sql_net":40,
    "time":41,
    "hostnames":42,
    "exec":43,
    "ntp_u":44,
    "discard":45,
    "nntp":46,
    "courier":47,
    "ctf":48,
    "ssh":49,
    "daytime":50,
    "shell":51,
    "netstat":52,
    "pop_3":53,
    "nnsp":54,
    "IRC":55,
    "pop_2":56,
    "printer":57,
    "tim_i":58,
    "pm_dump":59,
    "red_i":60,
    "netbios_ssn":61,
    "rje":62,
    "X11":63,
    "urh_i":64,
    "http_8001":65,
    "aol":66,
    "http_2784":67,
    "tftp_u":68,
    "harvest":69
    })

    kdd_dataset_test['protocol_type'] = kdd_dataset_test['protocol_type'].map({'tcp':0,'udp':1,'icmp':2})
    kdd_dataset_test['flag'] = kdd_dataset_test['flag'].map({'SF':0,'S1':1,'REJ':2,'S2':3,'S0':4,'S3':5,'RSTO':6,'RSTR':7,'RSTOS0':8,'OTH':9,'SH':10})

    #攻撃を数値に置換
    kdd_dataset_test['AttackTypes'] = kdd_dataset_test['AttackTypes'].map({
    "normal":0,
    "back":1,
    "land":1,
    "neptune":1,
    "smurf":1,
    "pod":1,
    "teardrop":1,
    "buffer_overflow":1,
    "loadmodule":1,
    "perl":1,
    "rootkit":1,
    "ftp_write":1,
    "guess_passwd":1,
    "imap":1,
    "multihop":1,
    "phf":1,
    "spy":1,
    "warezclient":1,
    "warezmaster":1,
    "satan":1,
    "ipsweep":1,
    "portsweep":1,
    "nmap": 1,

    "mscan": 1,
    "saint": 1,
    "udpstorm": 1,
    "apache2": 1,
    "processtable": 1,
    "mailbomb":1,
    "sqlattack": 1,
    "xterm": 1,
    "ps": 1,
    "httptunnel": 1,
    "xsnoop": 1,
    "xlock": 1,
    "worm": 1,
    "sendmail": 1,
    "named": 1,
    "snmpguess":1,
    "snmpgetattack": 1
    })

    #ラベルごとの枚数をカウントする
    print('TestDataCount-------------------------------------------------------------')
    num_eachAttack_test = kdd_dataset_test["AttackTypes"].value_counts()
    print(num_eachAttack_test)

# ##################################
# # ダミー変数化
# ##################################
    #TrainとTestを一旦連結
    train_num = len(kdd_dataset) #125973
    dataset = kdd_dataset.append(kdd_dataset_test) #dataset=kdd_dataset+kdd_dataset_test
    #csvファイルをpandasで読み込んだ時,要素が空白であると欠損値(NaN)(Not a Number)だと見なされてしまう
    dataset.fillna(0) #欠損値を0に穴埋めする

    data_dummy_1 = pd.get_dummies(dataset['protocol_type'],drop_first= False) #ダミー変数を作成
    dataset = pd.concat([dataset, data_dummy_1], axis=1) #ダミー変数を連結
    dataset = dataset.drop("protocol_type", axis=1) #ダミー変数の更新(列の削除)

    data_dummy_2 = pd.get_dummies(dataset['service'], drop_first = False)
    dataset = pd.concat([dataset, data_dummy_2], axis=1)
    dataset = dataset.drop("service", axis=1)

    data_dummy_3 = pd.get_dummies(dataset['flag'], drop_first = False)
    dataset = pd.concat([dataset, data_dummy_3], axis=1)
    dataset = dataset.drop("flag", axis=1)
    dataset.shape #[148517,123]

    #TrainとTestを再度分離
    kdd_dataset = dataset[:train_num] #一番最初からtrain_numまでを取る    
    kdd_test  = dataset[train_num:] #train_numから一番後ろまでを取る
    kdd_dataset.shape #[125973,123]
    kdd_test.shape #[22544,123]

# ##################################
# # ラベル情報の分離
# ##################################
    #attack_labelを分離
    attack_type = kdd_dataset['AttackTypes'] #攻撃を数値化したラベル
    kdd_dataset = kdd_dataset.drop("AttackTypes", axis=1) #AttackTypesの列を削除
    attack_type_test = kdd_test['AttackTypes']
    kdd_test = kdd_test.drop("AttackTypes", axis=1)

    #kdd_dataset,kdd_testの全数値を0〜1までに正規化
    from sklearn import preprocessing
    mm = preprocessing.MinMaxScaler()
    kdd_dataset = mm.fit_transform(kdd_dataset)
    mm = preprocessing.MinMaxScaler()
    kdd_test = mm.fit_transform(kdd_test)

    #chainerではfloat64は扱えない
    kdd_dataset = np.array(kdd_dataset,dtype='float32')
    kdd_test = np.array(kdd_test,dtype='float32')
    #攻撃を数値化したラベルを1次元ベクトル化
    kdd_label = np.ravel(attack_type)
    kdd_label_test = np.ravel(attack_type_test)
    #float64からint32へ変換（リストのラベルは整数のみのため）
    kdd_label = np.array(kdd_label,dtype='int32')
    kdd_label_test = np.array(kdd_label_test,dtype='int32')
    #攻撃を数値化したラベルを1次元ベクトル化
    kdd_type = np.ravel(attack_type)
    kdd_type_test = np.ravel(attack_type_test)
    #float64からint32へ変換（リストのラベルは整数のみのため）
    kdd_type = np.array(kdd_type,dtype='int32')
    kdd_type_test = np.array(kdd_type_test,dtype='int32')

    return kdd_label, kdd_dataset, kdd_type,kdd_label_test, kdd_test, kdd_type_test