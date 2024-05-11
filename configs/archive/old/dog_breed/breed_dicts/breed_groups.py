from collections import OrderedDict

__all__ = ["tsinghua_dict", "stanford_dict"]

general_dict = OrderedDict(
    {
        "french_bulldog": ["french_bulldog", "boston_bull"],
        "borzoi": [
            "saluki",
            "afghan_hound",
            "italian_greyhound",
            "borzoi",
            "whippet",
        ],
        "english_wolfhound": ["irish_wolfhound", "scottish_deerhound"],
        "french_shepherd": ["briard", "bouvier_des_flandres"],
        "komondor": ["komondor"],
        "short_wooled_middle_size_long_eared": [
            "basenji",
            "ibizan_hound",
            "african_hunting_dog",
        ],
        "bloodhound": [
            "bloodhound",
            "fila_braziliero",
            "black_and_tan_coonhound",
        ],
        "pug": ["pug", "brabancon_griffo"],
        "akita": ["dingo", "shiba_dog", "dhole"],
        "chow": ["chow"],
        "setter": ["gordon_setter", "english_setter", "irish_setter"],
        "beethoven": ["saint_bernard", "clumber", "newfoundland"],
        "bulldog": [
            "boxer",
            "american_staffordshire_terrier",
            "staffordshire_bullterrier",
        ],
        "pekinese": [
            "pekinese",
            "papillon",
            "chinese_rural_dog",
            "japanese_spaniel",
        ],
        "spitz": ["pomeranian", "japanese_spitzes", "keeshond"],
        "smooth_haired_big_size_lop_eared": [
            "rhodesian_ridgeback",
            "vizsla",
            "redbone",
            "curly_coated_retriever",
            "german_short_haired_pointer",
            "bluetick",
            "english_foxhound",
            "walker_hound",
            "weimaraner",
            "chesapeake_bay_retriever",
        ],
        "poodle": [
            "toy_poodle",
            "teddy",
            "bichon_frise",
            "old_english_sheepdog",
            "miniature_poodle",
            "irish_water_spaniel",
            "otterhound",
            "standard_poodle",
        ],
        "schnauzer": [
            "miniature_schnauzer",
            "standard_schnauzer",
            "giant_schnauzer",
        ],
        "english_terrier": [
            "irish_terrier",
            "airedale",
            "bedlington_terrier",
            "kerry_blue_terrier",
            "border_terrier",
            "lakeland_terrier",
            "wire_haired_fox_terrier",
        ],
        "shepherd": [
            "malinois",
            "groenendael",
            "german_shepherd",
            "leonberg",
            "black_sable",
            "norwegian_elkhound",
            "schipperke",
            "kelpie",
        ],
        "beagle_spaniel": [
            "brittany_spaniel",
            "basset",
            "english_springer",
            "welsh_springer_spaniel",
            "blenheim_spaniel",
            "sussex_spaniel",
            "beagle",
            "cocker_spaniel",
        ],
        "dog_doberman": ["doberman", "great_dane", "cane_carso"],
        "husky": [
            "eskimo_dog",
            "siberian_husky",
            "malamute",
            "samoyed",
        ],
        "labrador_retriever": [
            "flat_coated_retriever",
            "golden_retriever",
            "kuvasz",
            "labrador_retriever",
            "great_pyrenees",
        ],
        "york": [
            "west_highland_white_terrier",
            "australian_terrier",
            "yorkshire_terrier",
            "silky_terrier",
            "cairn",
            "norwich_terrier",
            "norfolk_terrier",
        ],
        "chinese_crested": ["chinese_crested_dog", "mexican_hairless"],
        "cardigan": ["cardigan", "pembroke"],
        "shih_tzu": [
            "shih_tzu",
            "maltese_dog",
            "lhasa",
            "tibetan_terrier",
            "affenpinscher",
        ],
        "collie": [
            "collie",
            "shetland_sheepdog",
            "border_collie",
            "australian_shepherd",
        ],
        "scotch_terrier": [
            "soft_coated_wheaten_terrier",
            "sealyham_terrier",
            "dandie_dinmont",
            "scotch_terrier",
        ],
        "rottweiler": ["rottweiler", "bull_mastiff", "tibetan_mastiff"],
        "swiss_mountain_dog": [
            "greater_swiss_mountain_dog",
            "bernese_mountain_dog",
            "entlebucher",
            "appenzeller",
        ],
        "toy_terrier": [
            "toy_terrier",
            "chihuahua",
            "miniature_pinscher",
        ],
    }
)

tsinghua_dict = OrderedDict(
    {
        "akita": [
            "217-n000079-dingo",
            "1043-n000001-Shiba_Dog",
            "209-n000043-dhole",
        ],
        "beagle_spaniel": [
            "232-n000076-Brittany_spaniel",
            "241-n000100-basset",
            "276-n000112-English_springer",
            "202-n000023-Welsh_springer_spaniel",
            "207-n000026-Blenheim_spaniel",
            "217-n000034-Sussex_spaniel",
            "276-n000113-beagle",
            "286-n000111-cocker_spaniel",
        ],
        "beethoven": [
            "211-n000068-Saint_Bernard",
            "235-n000097-clumber",
            "215-n000031-Newfoundland",
        ],
        "bloodhound": [
            "207-n000044-bloodhound",
            "225-n000062-black_and_tan_coonhound",
            "220-n000032-Fila Braziliero",
        ],
        "borzoi": [
            "206-n000051-Saluki",
            "222-n000013-Afghan_hound",
            "238-n000099-Italian_greyhound",
            "226-n000064-borzoi",
            "234-n000065-whippet",
        ],
        "bulldog": [
            "225-n000082-boxer",
            "216-n000078-American_Staffordshire_terrier",
            "203-n000015-Staffordshire_bullterrier",
        ],
        "cardigan": ["2909-n000116-Cardigan", "205-n000029-Pembroke"],
        "chinese_crested": [
            "234-n000093-Chinese_Crested_Dog",
            "220-n000069-Mexican_hairless",
        ],
        "chow": ["329-n000119-chow"],
        "collie": [
            "274-n000110-Shetland_sheepdog",
            "2594-n000109-Border_collie",
            "245-n000098-Australian_Shepherd",
            "227-n000094-collie",
        ],
        "dog_doberman": [
            "209-n000054-Doberman",
            "206-n000035-Great_Dane",
            "253-n000105-Cane_Carso",
        ],
        "english_terrier": [
            "207-n000036-Irish_terrier",
            "200-n000008-Airedale",
            "214-n000033-Bedlington_terrier",
            "229-n000087-Kerry_blue_terrier",
            "206-n000007-Border_terrier",
            "216-n000017-Lakeland_terrier",
            "226-n000057-wire_haired_fox_terrier",
        ],
        "english_wolfhound": [
            "224-n000039-Irish_wolfhound",
            "224-n000056-Scottish_deerhound",
        ],
        "french_bulldog": [
            "1121-n000002-French_bulldog",
            "230-n000084-Boston_bull",
        ],
        "french_shepherd": [
            "230-n000041-briard",
            "233-n000072-Bouvier_des_Flandres",
        ],
        "husky": [
            "232-n000083-Eskimo_dog",
            "1324-n000004-malamute",
            "2192-n000088-Samoyed",
            "1160-n000003-Siberian_husky",
        ],
        "komondor": ["318-n000115-komondor"],
        "labrador_retriever": [
            "206-n000047-flat_coated_retriever",
            "5355-n000126-golden_retriever",
            "223-n000067-kuvasz",
            "3580-n000122-Labrador_retriever",
            "225-n000073-Great_Pyrenees",
        ],
        "pekinese": [
            "480-n000125-Pekinese",
            "806-n000129-papillon",
            "3336-n000121-chinese_rural_dog",
            "249-n000106-Japanese_spaniel",
        ],
        "poodle": [
            "2925-n000114-toy_poodle",
            "7449-n000128-teddy",
            "3083-n000117-Bichon_Frise",
            "257-n000108-Old_English_sheepdog",
            "200-n000010-miniature_poodle",
            "219-n000066-Irish_water_spaniel",
            "217-n000014-otterhound",
            "316-n000118-standard_poodle",
        ],
        "pug": ["798-n000130-pug", "209-n000042-Brabancon_griffo"],
        "rottweiler": [
            "224-n000060-Rottweiler",
            "231-n000077-bull_mastiff",
            "205-n000030-Tibetan_mastiff",
        ],
        "schnauzer": [
            "2342-n000102-miniature_schnauzer",
            "220-n000053-standard_schnauzer",
            "232-n000089-giant_schnauzer",
        ],
        "scotch_terrier": [
            "235-n000090-soft_coated_wheaten_terrier",
            "211-n000058-Sealyham_terrier",
            "211-n000052-Dandie_Dinmont",
            "206-n000037-Scotch_terrier",
        ],
        "setter": [
            "228-n000085-Gordon_setter",
            "203-n000022-English_setter",
            "207-n000011-Irish_setter",
        ],
        "shepherd": [
            "233-n000071-malinois",
            "227-n000070-groenendael",
            "211-n000018-German_shepherd",
            "214-n000019-Leonberg",
            "258-n000104-Black_sable",
            "217-n000046-Norwegian_elkhound",
            "201-n000024-schipperke",
            "209-n000049-kelpie",
        ],
        "shih_tzu": [
            "361-n000123-Shih_Tzu",
            "249-n000101-Maltese_dog",
            "215-n000038-Lhasa",
            "203-n000021-Tibetan_terrier",
            "200-n000012-affenpinscher",
        ],
        "short_wooled_middle_size_long_eared": [
            "243-n000103-basenji",
            "210-n000006-Ibizan_hound",
            "211-n000025-African_hunting_dog",
        ],
        "smooth_haired_big_size_lop_eared": [
            "230-n000080-Rhodesian_ridgeback",
            "210-n000048-vizsla",
            "207-n000045-redbone",
            "202-n000028-curly_coated_retriever",
            "211-n000059-German_short_haired_pointer",
            "217-n000050-bluetick",
            "223-n000074-English_foxhound",
            "216-n000063-Walker_hound",
            "235-n000096-Weimaraner",
            "215-n000075-Chesapeake_Bay_retriever",
        ],
        "spitz": [
            "1936-n000005-Pomeranian",
            "253-n000107-Japanese_Spitzes",
            "223-n000092-keeshond",
        ],
        "swiss_mountain_dog": [
            "237-n000086-Greater_Swiss_Mountain_dog",
            "211-n000061-Bernese_mountain_dog",
            "221-n000055-EntleBucher",
            "234-n000091-Appenzeller",
        ],
        "toy_terrier": [
            "237-n000095-toy_terrier",
            "420-n000124-Chihuahua",
            "561-n000127-miniature_pinscher",
        ],
        "york": [
            "209-n000040-West_Highland_white_terrier",
            "202-n000020-Australian_terrier",
            "340-n000120-Yorkshire_terrier",
            "231-n000081-silky_terrier",
            "211-n000009-cairn",
            "203-n000016-Norwich_terrier",
            "203-n000027-Norfolk_terrier",
        ],
    }
)

stanford_dict = OrderedDict(
    {
        "akita": ["n02115641-dingo", "n02115913-dhole"],
        "beagle_spaniel": [
            "n02101388-Brittany_spaniel",
            "n02088238-basset",
            "n02102040-English_springer",
            "n02102177-Welsh_springer_spaniel",
            "n02086646-Blenheim_spaniel",
            "n02102480-Sussex_spaniel",
            "n02088364-beagle",
            "n02102318-cocker_spaniel",
        ],
        "beethoven": [
            "n02109525-Saint_Bernard",
            "n02101556-clumber",
            "n02111277-Newfoundland",
        ],
        "bloodhound": [
            "n02088466-bloodhound",
            "n02089078-black-and-tan_coonhound",
        ],
        "borzoi": [
            "n02091831-Saluki",
            "n02088094-Afghan_hound",
            "n02091032-Italian_greyhound",
            "n02090622-borzoi",
            "n02091134-whippet",
        ],
        "bulldog": [
            "n02108089-boxer",
            "n02093428-American_Staffordshire_terrier",
            "n02093256-Staffordshire_bullterrier",
        ],
        "cardigan": ["n02113186-Cardigan", "n02113023-Pembroke"],
        "chinese_crested": ["n02113978-Mexican_hairless"],
        "chow": ["n02112137-chow"],
        "collie": [
            "n02105855-Shetland_sheepdog",
            "n02106166-Border_collie",
            "n02106030-collie",
        ],
        "dog_doberman": ["n02107142-Doberman", "n02109047-Great_Dane"],
        "english_terrier": [
            "n02093991-Irish_terrier",
            "n02096051-Airedale",
            "n02093647-Bedlington_terrier",
            "n02093859-Kerry_blue_terrier",
            "n02093754-Border_terrier",
            "n02095570-Lakeland_terrier",
            "n02095314-wire-haired_fox_terrier",
        ],
        "english_wolfhound": [
            "n02090721-Irish_wolfhound",
            "n02092002-Scottish_deerhound",
        ],
        "french_bulldog": [
            "n02108915-French_bulldog",
            "n02096585-Boston_bull",
        ],
        "french_shepherd": [
            "n02105251-briard",
            "n02106382-Bouvier_des_Flandres",
        ],
        "husky": [
            "n02109961-Eskimo_dog",
            "n02110063-malamute",
            "n02111889-Samoyed",
            "n02110185-Siberian_husky",
        ],
        "komondor": ["n02105505-komondor"],
        "labrador_retriever": [
            "n02099601-golden_retriever",
            "n02104029-kuvasz",
            "n02099712-Labrador_retriever",
            "n02111500-Great_Pyrenees",
            "n02099267-flat-coated_retriever",
        ],
        "pekinese": [
            "n02086079-Pekinese",
            "n02086910-papillon",
            "n02085782-Japanese_spaniel",
        ],
        "poodle": [
            "n02113624-toy_poodle",
            "n02105641-Old_English_sheepdog",
            "n02113712-miniature_poodle",
            "n02102973-Irish_water_spaniel",
            "n02091635-otterhound",
            "n02113799-standard_poodle",
        ],
        "pug": ["n02110958-pug", "n02112706-Brabancon_griffon"],
        "rottweiler": [
            "n02106550-Rottweiler",
            "n02108422-bull_mastiff",
            "n02108551-Tibetan_mastiff",
        ],
        "schnauzer": [
            "n02097047-miniature_schnauzer",
            "n02097209-standard_schnauzer",
            "n02097130-giant_schnauzer",
        ],
        "scotch_terrier": [
            "n02095889-Sealyham_terrier",
            "n02096437-Dandie_Dinmont",
            "n02097298-Scotch_terrier",
            "n02098105-soft-coated_wheaten_terrier",
        ],
        "setter": [
            "n02101006-Gordon_setter",
            "n02100735-English_setter",
            "n02100877-Irish_setter",
        ],
        "shepherd": [
            "n02105162-malinois",
            "n02105056-groenendael",
            "n02106662-German_shepherd",
            "n02111129-Leonberg",
            "n02091467-Norwegian_elkhound",
            "n02104365-schipperke",
            "n02105412-kelpie",
        ],
        "shih_tzu": [
            "n02085936-Maltese_dog",
            "n02098413-Lhasa",
            "n02097474-Tibetan_terrier",
            "n02110627-affenpinscher",
            "n02086240-Shih-Tzu",
        ],
        "short_wooled_middle_size_long_eared": [
            "n02110806-basenji",
            "n02091244-Ibizan_hound",
            "n02116738-African_hunting_dog",
        ],
        "smooth_haired_big_size_lop_eared": [
            "n02087394-Rhodesian_ridgeback",
            "n02100583-vizsla",
            "n02090379-redbone",
            "n02088632-bluetick",
            "n02089973-English_foxhound",
            "n02089867-Walker_hound",
            "n02092339-Weimaraner",
            "n02099849-Chesapeake_Bay_retriever",
            "n02099429-curly-coated_retriever",
            "n02100236-German_short-haired_pointer",
        ],
        "spitz": ["n02112018-Pomeranian", "n02112350-keeshond"],
        "swiss_mountain_dog": [
            "n02107574-Greater_Swiss_Mountain_dog",
            "n02107683-Bernese_mountain_dog",
            "n02108000-EntleBucher",
            "n02107908-Appenzeller",
        ],
        "toy_terrier": [
            "n02087046-toy_terrier",
            "n02085620-Chihuahua",
            "n02107312-miniature_pinscher",
        ],
        "york": [
            "n02098286-West_Highland_white_terrier",
            "n02096294-Australian_terrier",
            "n02094433-Yorkshire_terrier",
            "n02097658-silky_terrier",
            "n02096177-cairn",
            "n02094258-Norwich_terrier",
            "n02094114-Norfolk_terrier",
        ],
    }
)
