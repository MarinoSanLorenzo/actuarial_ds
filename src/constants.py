
class Constants:
    URL_LINK_TO_DATA = 'URL_LINK_TO_DATA'
    EXPOSURE_NAME = 'EXPOSURE_NAME'
    NB_CLAIMS = 'NB_CLAIMS'
    CLAIM_AMOUNT = 'CLAIM_AMOUNT'
    CLAIM_FREQUENCY = 'CLAIM_FREQUENCY'
    VARIABLES_TO_EXCLUDE = 'VARIABLES_TO_EXCLUDE'
    MAX_NB_ANCIENNETE = 'MAX_NB_ANCIENNETE'
    MAX_CLAIM_AMOUNT = 'MAX_CLAIM_AMOUNT'
    TEST_SIZE = 'TEST_SIZE'
    RANDOM_STATE = 'RANDOM_STATE'
    LINK_TO_POSTAL_CODE_MAPPING = 'LINK_TO_POSTAL_CODE_MAPPING'


params = {
    Constants.URL_LINK_TO_DATA: "https://gitfront.io/r/katrienantonio/a29071bdc7b2f20f24268be573ef54b68c65168c/IABE-DS-module-1/raw/assignment/assignment_data.csv",
    Constants.EXPOSURE_NAME:'duree',
    Constants.CLAIM_FREQUENCY:'claim_frequency',
    Constants.NB_CLAIMS:'nbrtotc',
    Constants.CLAIM_AMOUNT:'chargtot',
    Constants.VARIABLES_TO_EXCLUDE:['lnexpo'],
    Constants.MAX_NB_ANCIENNETE: 82,
    Constants.MAX_CLAIM_AMOUNT: 500_000,
    Constants.TEST_SIZE:0.2,
    Constants.RANDOM_STATE:42,
    Constants.LINK_TO_POSTAL_CODE_MAPPING: './postal_code_mapping.csv'

}