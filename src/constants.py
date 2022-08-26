import os



class Constants2:
    PATH_TO_DATA = 'PATH_TO_DATA'
    NB_CLAIMS = "NB_CLAIMS"
    CLAIM_AMOUNT = "CLAIM_AMOUNT"
    CLAIM_FREQUENCY = "CLAIM_FREQUENCY"
    EXPOSURE_NAME = "EXPOSURE_NAME"
    VARIABLES_TO_EXCLUDE = "VARIABLES_TO_EXCLUDE"
    RENAMING_DUMMY_CODING_MAPPING = "RENAMING_DUMMY_CODING_MAPPING"


params_blog_2 = {
    Constants2.PATH_TO_DATA:"./data",
    Constants2.NB_CLAIMS:'ClaimNb',
    Constants2.CLAIM_AMOUNT:'ClaimAmount',
    Constants2.EXPOSURE_NAME:'Exposure',
    Constants2.VARIABLES_TO_EXCLUDE: ["PolicyID"],
    Constants2.CLAIM_FREQUENCY:"claim_frequency",
    Constants2.RENAMING_DUMMY_CODING_MAPPING : {
                            "ClaimNb":"ClaimNb",
                            "Exposure":"Exposure",
                            "ClaimAmount":"ClaimAmount",
                            "claim_frequency":"claim_frequency",
                            "Power_d":"power_d",
                            "Power_e":"power_e",
                            "Power_f":"power_f",
                            "Power_g":"power_g",
                            "Power_h":"power_h",
                            "Power_i":"power_i",
                            "Power_j":"power_j",
                            "Power_k":"power_k",
                            "Power_l":"power_l",
                            "Power_n":"power_n",
                            "Power_o":"power_o",
                            "Brand_Fiat":"brand_fiat",
                            "Brand_Mercedes, Chrysler or BMW":"brand_mercedes_chrysler_bmw",
                            "Brand_Opel, General Motors or Ford":"brand_opel_generalmotors_ford",
                            "Brand_Renault, Nissan or Citroen":"brand_renault_nissan_citroen",
                            "Brand_Volkswagen, Audi, Skoda or Seat":"brand_volkswagen_audi_skoda_seat",
                            "Brand_other":"brand_other",
                            "Gas_Regular":"gas_regular",
                            "Region_Aquitaine":"region_aquitaine",
                            "Region_Basse-Normandie":"region_basse_normandie",
                            "Region_Bretagne":"region_bretagne",
                            "Region_Centre":"region_centre",
                            "Region_Ile-de-France":"region_ile_de_france",
                            "Region_Limousin":"region_limousin",
                            "Region_Nord-Pas-de-Calais":"region_nord_pas_de_calais",
                            "Region_Pays-de-la-Loire":"region_pays_de_la_loire",
                            "Region_Poitou-Charentes":"region_poitou_charentes",
                            # "DriverAge_bin_(17.999, 32.0]":"driverage_bin_eighteen_thirty_two",
                            "DriverAge_bin_(17.999, 32.0]":"driverage_bin_18_to_32",
                            "DriverAge_bin_(32.0, 40.0]":"driverage_bin_32_to_40",
                            # "DriverAge_bin_(32.0, 40.0]":"driverage_bin_thirty_two_forty",
                            "DriverAge_bin_(40.0, 48.0]":"driverage_bin_40_to_48",
                            # "DriverAge_bin_(40.0, 48.0]":"driverage_bin_forty_forty_eight",
                            "DriverAge_bin_(48.0, 57.0]":"driverage_bin_48_to_57",
                            "DriverAge_bin_(57.0, 99.0]":"driverage_bin_57_to_99",
                            # "DriverAge_bin_(48.0, 57.0]":"driverage_bin_forty_eight_fifty_seven",
                            "CarAge_bin_(0.0, 2.0]":"carage_bin_0_to_2",
                            "CarAge_bin_(2.0, 5.0]":"carage_bin_2_to_5",
                            "CarAge_bin_(5.0, 9.0]":"carage_bin_5_to_9",
                            "CarAge_bin_(9.0, 13.0]":"carage_bin_9_to_13",
                            "CarAge_bin_(13.0, 100.0]":"carage_bin_13_to_100",
                            "Density_bin_(1.999, 51.0]":"density_bin_2_to_51",
                            "Density_bin_(51.0, 158.0]":"density_bin_51_to_150",
                            "Density_bin_(158.0, 555.0]":"density_bin_150_to_555",
                            "Density_bin_(555.0, 2404.0]":"density_bin_555_to_2404",
                            "Density_bin_(2404.0, 27000.0]":"density_bin_2404_27000",
    }
}


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

