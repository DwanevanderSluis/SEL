import requests
url = "https://swat.d4science.org/tag"
document = {
"title": "Obama travels.",
"headline": "A toy example.",
"content": "Barack Obama was in Pisa for a flying visit.", "mentions": "proper,common"
}
requests.post(url, document, params={"gcube-token": MY_GCUBE_TOKEN})