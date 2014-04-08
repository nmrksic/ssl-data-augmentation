names is the list of film names, sparse matrix is a film x genre binary matrix. To be used with bayes, we need sparseMatrix', and names is used to label entries after sorting by likelihood. A very good cluster is the set of George Lucas/SF films: 

lucas

indices : 172 181 210


results: 
181	9.58508	181|Return of the Jedi (1983)
50	9.58508	50|Star Wars (1977)
172	9.34084	172|Empire Strikes Back, The (1980)
271	7.99097	271|Starship Troopers (1997)
498	7.14069	498|African Queen, The (1951)
897	5.20461	897|Time Tracers (1995)
450	5.20461	450|Star Trek V: The Final Frontier (1989)
449	5.20461	449|Star Trek: The Motion Picture (1979)
380	5.20461	380|Star Trek: Generations (1994)
373	5.20461	373|Judge Dredd (1995)
230	5.20461	230|Star Trek IV: The Voyage Home (1986)
229	5.20461	229|Star Trek III: The Search for Spock (1984)
228	5.20461	228|Star Trek: The Wrath of Khan (1982)
227	5.20461	227|Star Trek VI: The Undiscovered Country (1991)
222	5.20461	222|Star Trek: First Contact (1996)
82	5.20461	82|Jurassic Park (1993)
62	5.20461	62|Stargate (1994)
121	5.01092	121|Independence Day (ID4)
110	4.40121	110|Operation Dumbo Drop (1995)
1239	4.35433	1239|Cutthroat Island (1995)



or al pacino, but not as good: 

>> q = [127 187 307 273 ];
>> s = BayesianSet(sparseMatrix', q);
>> bsshow(s, 'names.txt', 20);
307	6.02622	307|Devil's Advocate, The (1997)
334	5.33616	334|U Turn (1997)
17	4.86122	17|From Dusk Till Dawn (1996)
914	4.84563	914|Wild Things (1998)
1139	4.8452	1139|Hackers (1995)
273	4.8452	273|Heat (1995)
1586	3.76665	1586|Lashou shentan (1992)
1478	3.76665	1478|Dead Presidents (1995)
806	3.76665	806|Menace II Society (1993)
187	3.76665	187|Godfather: Part II, The (1974)
127	3.76665	127|Godfather, The (1972)
802	3.69004	802|Hard Target (1993)
1669	3.58147	1669|MURDER and murder (1996)
327	3.58147	327|Cop Land (1997)
1618	3.58104	1618|King of New York (1990)
1304	3.58104	1304|New York Cop (1996)
1277	3.58104	1277|Set It Off (1996)
770	3.5518	770|Devil in a Blue Dress (1995)
302	3.5518	302|L.A. Confidential (1997)
902	3.30984	902|Big Lebowski, The (1998)


A very bad one (to be checked!) is Tarantino: 

56 156 3 92 943

