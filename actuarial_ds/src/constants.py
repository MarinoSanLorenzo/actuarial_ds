
class Constants:
    URL_LINK_TO_DATA = 'URL_LINK_TO_DATA'
    EXPOSURE_NAME = 'EXPOSURE_NAME'
    NB_CLAIMS = 'NB_CLAIMS'
    CLAIM_AMOUNT = 'CLAIM_AMOUNT'
    VARIABLES_TO_EXCLUDE = 'VARIABLES_TO_EXCLUDE'
    MAX_NB_ANCIENNETE = 'MAX_NB_ANCIENNETE'
    MAX_CLAIM_AMOUNT = 'MAX_CLAIM_AMOUNT'


params = {
    Constants.URL_LINK_TO_DATA: "https://gitfront.io/r/katrienantonio/a29071bdc7b2f20f24268be573ef54b68c65168c/IABE-DS-module-1/raw/assignment/assignment_data.csv",
    Constants.EXPOSURE_NAME:'duree',
    Constants.NB_CLAIMS:'nbrtotc',
    Constants.CLAIM_AMOUNT:'chargtot',
    Constants.VARIABLES_TO_EXCLUDE:['lnexpo'],
    Constants.MAX_NB_ANCIENNETE: 82,
    Constants.MAX_CLAIM_AMOUNT: 500_000,

}