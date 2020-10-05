**I. Entraînement d'un réseau de neurones**

Le code proposé ici est issu du dépôt suivant https://github.com/Amustache/ML-2019.
Il a été partiellement modifié.

**A. Installation**

Nous avons utilisé les librairies suivantes : 
- python 3.8.5
- tensorflow 2.3.0
- matplotlib 3.3.2
- opencv-python 4.4.0
- opencv-contrib-python 4.4.0
- numpy 1.19.2
- tqdm 4.49.0
- pandas 1.1.2
- sklearn 0.23.1
- cuda 11.0.3

**B. Structure des données**

- Les images doivent être redimensionnées en 640x480.
- Nous avons utilisé LabelImg (https://github.com/tzutalin/labelImg) pour labeliser les images. Nous obtenons des fichiers xml, que nous plaçons dans le même dossier que les images associées
- Pour le bon fonctionnement du code, utilisez l'arborescence suivante :
		
		dir/
			data/
				in/
				out/
			scripts/
			

**C. Entraînement**

Utilisez le script train.py.

****
**II. Les bases de données**

Nous avons quatre bases de données :

**A. bdd_annonces**

Cette base de données correspond au dépouillement des publicités de *Joystick* de janvier 1988 à décembre 1998. Nous y indiquons :

- L'identifiant de la publicité.
- Le titre, afin de distinguer *Joystick* et Joystick Hebdo.
-  Le numéro.
-  La date de publication. Notons que les doubles numéros d'été sont datés uniquement du mois de juillet.
- Le nombre de pages total du numéro.
- La ou les pages contenant la publicité, avec une virgule comme séparateur si besoin.
- La colonne "identique" indique si une même publicité est présente dans plusieurs numéros. Les identifiants des publicités correspondantes sont renseignés.
- Le nom du ou des produits annoncés, avec une virgule comme séparateur si besoin.
-  Le type de produit annoncé, avec une virgule comme séparateur si besoin.
- Le nom du ou des annonceurs. Nous avons fait le choix de relever un maximum d'entités citées dans une même publicité afin de permettre des études futures sur les informations et les relations qu'elles comportent. Une virgule sert de séparateur.
- L'adresse de l'annonceur, du moins celle indiquée sur une publicité. Pour les enseignes avec plusieurs magasins nous avons choisi l'adresse principale (VPC, magasin historique).
- Le type d'annonceur (captif, hors captif ou le magazine).
- Le format de la publicité (pleine page encart, encart double, et double pleine page).
- La colonne "suivi" indique si la publicité fait partie d'un ensemble composé de plusieurs publicité au message publicitaire dilué sur l'ensemble. Dans ce cas, les identifiants des autres publicités sont indiquées.
-  Enfin, nous relevons si la publicité est principalement graphique ou textuelle. Avec du recul, nous jugeons que cette information est peu pertinente car trop subjective.

**B. bdd_nb_annonces**

Cette base synthétise les informations sur le nombre de publicités. Nous y indiquons :

- L'identifiant du numéro.
- Le numéro.
- Le titre du magazine.
- Le nombre de publicités dans le titre.
- Le nombre de pages dans le titre.
- La date du numéro.

**C. bdd_frames**

Cette base a pour but d'aider à une identification des "cadres" visuels des publicités. Il s'agit de publicités quasi-identiques, mais dont un élément change (souvent du texte, dans le cas des *listings* de jeux par exemple). Elles comportent ainsi un cadre visuel fixe et des éléments secondaires changeant. Nous y indiquons :

- L'identifiant du cadre.
- Le nom du produit.
- Le nom de l'annonceur, avec une virgule comme séparateur en cas de multiples noms relevés.
- Les identifiants des publicités concernées issus de bdd_annonces.

**C. bbd_mags**

Cette base comporte les informations générales sur les magazines du corpus. Nous y indiquons :

- L'identifiant.
- Le titre du magazine.
- La date du premier numéro.
- La date du dernier numéro.
- La durée de publication en années.
- Le ou les sociétés éditrices du titres.
- Le type de publication (micro/PC, consoles, généraliste).
- Le nombre total de numéros identifiés.
- Le nombre total de numéros identifiés pour la période concernée.
- Le nombre total de numéros numérisés pour la période concernée.
- Le prix du premier numéro.
- Le prix du dernier numéro.
- La périodicité.


       

