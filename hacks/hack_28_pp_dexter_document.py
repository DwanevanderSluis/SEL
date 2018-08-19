import pprint
import json

article = json.loads('{"docId": 28, "title": "Sudan\u2019s Vice President killed in helicopter crash", "saliency": [{"entityid": 31816, "score": 1.4}, {"entityid": 19541, "score": 1.8}, {"entityid": 27421, "score": 2.4}, {"entityid": 676659, "score": 1.2}, {"entityid": 25645, "score": 1.2}, {"entityid": 1140170, "score": 2.6}, {"entityid": 605837, "score": 1.4}, {"entityid": 18337522, "score": 1.6}, {"entityid": 17320, "score": 1.6}, {"entityid": 32350676, "score": 1.8}, {"entityid": 683270, "score": 1.6}], "timestamp": "2011-07-10T19:05:58Z", "wikiTitle": "Sudan\u2019s_Vice_President_killed_in_helicopter_crash", "wikinewsId": 17315, "document": [{"name": "headline", "value": "Sudan\u2019s Vice President killed in helicopter crash"}, {"name": "dateline", "value": "August 1, 2005"}, {"name": "body_par_000", "value": "John Garang, recently sworn in as Vice President of the Sudan, was confirmed dead after his helicopter disappeared yesterday. Sudanese state television initially reported him to be alive but it now appears Mr. Garang died after his helicopter crashed."}, {"name": "body_par_001", "value": "Garang became Vice President in a historic deal, which ended the southern rebellion after 20 years of fighting. The deal created a government of national unity, giving hope to many that problems endemic to the region, including the Janjaweed militia in Darfur, could finally be solved and that the Muslim north could come to liveable terms with the Christian south."}, {"name": "body_par_002", "value": "Garang was making an official trip to Uganda when bad weather struck; his helicopter had not arrived at its destination and was out of contact leading many to fear the worst."}, {"name": "body_par_003", "value": "The BBC reports that \"large-scale\" rioting has broken out in the Sudanese capital Khartoum, with supporters of Mr. Garang battling armed police. Disturbances are also reported in the south of the country. Some southerners believe that the Sudanese authorities were behind the crash, according to USA Today, which draws parallels with the shooting down of a plane carrying Rwandas President Habyarimana in 1994. Habyarimanas assassination helped trigger Rwandas infamous genocide, recently portrayed in the movie \"Hotel Rwanda\"."}, {"name": "body_par_004", "value": "Garangs party, the Sudan Peoples Liberation Movement, has appealed for calm, insisting that his death was an accident."}, {"name": "body_par_005", "value": "In a press release published on the website Allafrica.com, Ugandas President Museveni appeared to be keeping an open mind about the circumstances of the crash, which he says took place on the Ugandan-Sudanese border. \"The helicopter was a recently overhauled executive helicopter that has served us well for the last 8 Years\", he is quoted as saying, detailing a number of technical improvements recently made to the craft. In the same statement, Museveni announces the creation of a panel of experts to investigate the crash. \"We have also approached a certain foreign government to rule out any form of sabotage or terrorism\", he says."}]}')

pprint.pprint(article)