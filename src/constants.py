class Constants:
    URL_LINK_TO_DATA = "URL_LINK_TO_DATA"
    EXPOSURE_NAME = "EXPOSURE_NAME"
    NB_CLAIMS = "NB_CLAIMS"
    CLAIM_AMOUNT = "CLAIM_AMOUNT"
    CLAIM_FREQUENCY = "CLAIM_FREQUENCY"
    VARIABLES_TO_EXCLUDE = "VARIABLES_TO_EXCLUDE"
    MAX_NB_ANCIENNETE = "MAX_NB_ANCIENNETE"
    MAX_CLAIM_AMOUNT = "MAX_CLAIM_AMOUNT"
    TEST_SIZE = "TEST_SIZE"
    RANDOM_STATE = "RANDOM_STATE"
    LINK_TO_POSTAL_CODE_MAPPING = "LINK_TO_POSTAL_CODE_MAPPING"
    RENAMING_DUMMY_CODING_MAPPING = "RENAMING_DUMMY_CODING_MAPPING"


params = {
    Constants.URL_LINK_TO_DATA: "https://gitfront.io/r/katrienantonio/a29071bdc7b2f20f24268be573ef54b68c65168c/IABE-DS-module-1/raw/assignment/assignment_data.csv",
    Constants.EXPOSURE_NAME: "duree",
    Constants.CLAIM_FREQUENCY: "claim_frequency",
    Constants.NB_CLAIMS: "nbrtotc",
    Constants.CLAIM_AMOUNT: "chargtot",
    Constants.VARIABLES_TO_EXCLUDE: ["lnexpo"],
    Constants.MAX_NB_ANCIENNETE: 82,
    Constants.MAX_CLAIM_AMOUNT: 500_000,
    Constants.TEST_SIZE: 0.2,
    Constants.RANDOM_STATE: 42,
    Constants.LINK_TO_POSTAL_CODE_MAPPING: "./postal_code_mapping.csv",  # source https://www.odwb.be/explore/dataset/code-postaux-belge/export/
    Constants.RENAMING_DUMMY_CODING_MAPPING: {
        "duree": "duree",
        "nbrtotc": "nbrtotc",
        "chargtot": "chargtot",
        "agecar_0-1": "agecar_zero_one",
        "agecar_2-5": "agecar_two_five",
        "agecar_6-10": "agecar_six_ten",
        "agecar_>10": "agecar_higher_ten",
        "sexp_Female": "sexp_Female",
        "sexp_Male": "sexp_Male",
        "fuelc_Gasoil": "fuelc_Gasoil",
        "fuelc_Petrol": "fuelc_Petrol",
        "split_Monthly": "split_Monthly",
        "split_Once": "split_Once",
        "split_Thrice": "split_Thrice",
        "split_Twice": "split_Twice",
        "usec_Private": "usec_Private",
        "usec_Professional": "usec_Professional",
        "fleetc_No": "fleetc_No",
        "fleetc_Yes": "fleetc_Yes",
        "sportc_No": "sportc_No",
        "sportc_Yes": "sportc_Yes",
        "coverp_MTPL": "coverp_MTPL",
        "coverp_MTPL+": "coverp_MTPLplus",
        "coverp_MTPL+++": "coverp_MTPLplusplusplus",
        "powerc_66-110": "powerc_sixtysix_onehundredten",
        "powerc_<66": "powerc_higher_sixtysix",
        "powerc_>110": "powerc_higher_onehundredten",
        "district_risk_group_high": "district_risk_group_high",
        "district_risk_group_low": "district_risk_group_low",
        "district_risk_group_medium": "district_risk_group_medium",
        "AGEPH_bin_(16.999, 33.0]": "AGEPH_bin_seventeen_thirtythree",
        "AGEPH_bin_(33.0, 41.0]": "AGEPH_bin_thirtythree_forty_one",
        "AGEPH_bin_(41.0, 50.0]": "AGEPH_bin_forty_one_fifty",
        "AGEPH_bin_(50.0, 61.0]": "AGEPH_bin_fifty_sixty_one",
        "AGEPH_bin_(61.0, 95.0]": "AGEPH_bin_sixty_one_ninety_five",
    },
}
