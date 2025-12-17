## Livrable éthique – Démarche adoptée à partir d’ALTAI (Commission Européenne)

### Introduction
Pour structurer notre démarche éthique, nous nous sommes appuyés sur l’Assessment List for Trustworthy AI (ALTAI), publié par le groupe d’experts mis en place par la Commission Européenne. ALTAI est présenté comme un outil d’auto-évaluation, pensé pour être utilisé de manière flexible et adapté au contexte du projet, avec un objectif très concret : comprendre les risques qu’un système d’IA peut générer et définir des moyens réalistes pour les réduire, tout en maximisant les bénéfices. :contentReference[oaicite:0]{index=0}

Le document recommande aussi de ne pas faire cet exercice “en solo”, mais de le compléter avec une équipe pluridisciplinaire (technique, métier, conformité, management, utilisateurs). Dans notre projet, on a gardé cette logique : les choix sur les données, les transformations, le cadre d’usage et la manière de présenter les résultats ont été discutés collectivement, en tenant compte des impacts possibles sur les employés et sur l’organisation. :contentReference[oaicite:1]{index=1}

Enfin, ALTAI rappelle que l’auto-évaluation doit être ancrée dans la protection des droits fondamentaux et propose de commencer par une analyse d’impact sur les droits fondamentaux (FRIA) avec des questions sur la discrimination potentielle et la protection des données personnelles. C’est particulièrement pertinent dans un contexte RH : même un “bon” modèle peut avoir des effets injustes ou intrusifs si son usage n’est pas cadré. :contentReference[oaicite:2]{index=2}

### 1. Respect de l’autonomie humaine
ALTAI insiste sur le fait que les systèmes qui guident ou soutiennent la décision, notamment les systèmes de prédiction de risque, peuvent influencer les comportements humains, la confiance, et créer de la dépendance ou de la confusion sur l’origine d’une décision. Dans notre cas, le modèle d’attrition entre exactement dans cette catégorie : il produit un score qui peut orienter des décisions RH ou managériales. :contentReference[oaicite:3]{index=3}

La décision centrale prise en équipe a donc été de cadrer le modèle comme une aide à la décision, et jamais comme une décision automatique. Concrètement, cela signifie que le score sert à déclencher une discussion ou une analyse, pas une action “mécanique”. L’objectif attendu n’est pas de sanctionner, mais d’anticiper des actions positives d’accompagnement (écoute, amélioration des conditions de travail, formation, clarification des perspectives). On a aussi considéré le risque de surconfiance : si un manager voit un score, il peut lui donner plus de poids qu’il ne devrait. Pour réduire ce risque, ALTAI recommande des dispositifs d’oversight et de formation des personnes qui supervisent ou utilisent le système. :contentReference[oaicite:4]{index=4}

En nous appuyant sur les notions d’human-in-command décrites par ALTAI, nous proposons que les RH et managers gardent la capacité de choisir quand utiliser le score, de ne pas l’utiliser dans certains cas, et de pouvoir contredire le modèle lorsque le contexte humain le justifie. 

### 2. Robustesse technique et sécurité
ALTAI rappelle que la robustesse ne se limite pas à “un bon score”, mais inclut la sécurité, la fiabilité, la reproductibilité, et surtout la préparation à l’échec. L’outil invite à réfléchir à ce qui se passe si le système est moins précis que prévu, si les données changent, ou si des attaques ou manipulations sont possibles. :contentReference[oaicite:6]{index=6}

Dans le projet, notre démarche a été de tester la stabilité avec de la validation croisée et d’utiliser plusieurs métriques plutôt que de se limiter à l’accuracy, parce que dans un problème d’attrition les faux positifs et les faux négatifs n’ont pas le même coût. ALTAI souligne d’ailleurs qu’il faut surveiller des mesures comme les faux positifs, faux négatifs ou le F1-score pour éviter de se tromper sur la performance réelle, et surtout communiquer ces limites correctement aux utilisateurs pour éviter des attentes irréalistes. :contentReference[oaicite:7]{index=7}

Un point important recommandé par ALTAI est l’existence de plans de repli (fallback) et de procédures en cas de faible confiance ou d’erreur. Dans notre contexte, cela se traduit par une règle simple : si le modèle est instable, si la qualité de données se dégrade, ou si les performances baissent, on suspend l’usage opérationnel et on revient à une analyse RH classique le temps de réévaluer. 

### 3. Confidentialité et gouvernance des données
ALTAI met la confidentialité au cœur de la prévention des risques, en rappelant la logique privacy-by-design et privacy-by-default et l’importance de mesures comme le chiffrement, la pseudonymisation, l’agrégation, l’anonymisation, la limitation d’accès et la journalisation des accès. Il mentionne aussi la nécessité d’évaluer si une analyse d’impact relative à la protection des données (DPIA) est requise selon le niveau de risque et le type de traitement. 

Dans notre projet, la discussion la plus importante a porté sur la badgeuse. Le risque éthique évident est de transformer un outil de présence en outil de surveillance. La décision retenue est de privilégier des indicateurs agrégés et utiles à l’analyse, plutôt que des données fines qui pourraient permettre un suivi trop intrusif. Cette décision s’aligne avec les notions d’agrégation et de minimisation évoquées par ALTAI. 

Enfin, nous proposons un cadre de gouvernance clair : accès restreint aux résultats individuels, traçabilité des consultations, et séparation entre les besoins d’analyse et les besoins d’action. L’idée est de réduire la surface de risque, pas seulement de “sécuriser un fichier”.

### 4. Transparence
ALTAI définit la transparence comme un ensemble qui combine la traçabilité, l’explicabilité et une communication ouverte sur les limites du système. Ce point est crucial dans un projet RH, parce qu’un score peut être mal compris, surinterprété, ou utilisé comme argument d’autorité. 

Côté traçabilité, ALTAI encourage à pouvoir retracer quelles données ont été utilisées, quel modèle a produit quelle recommandation, et à mettre en place des pratiques de logging. Nous avons donc documenté le pipeline, les transformations, et les versions du modèle. :contentReference[oaicite:12]{index=12}

Côté explicabilité, ALTAI rappelle que l’objectif est de rendre le système intelligible pour des non-experts, et que l’explication permet aussi, lorsque c’est nécessaire, de contester une décision influencée par l’IA. Dans notre projet, cela se traduit par une volonté de présenter les résultats avec des éléments pédagogiques (sens des métriques, types d’erreurs) et des facteurs influents compréhensibles, en restant prudents sur ce que le modèle peut réellement expliquer. 

Enfin, ALTAI insiste sur la communication : expliquer le but, les critères et les limites, et communiquer les taux d’erreur de manière adaptée. Notre choix est donc d’éviter les formulations qui donnent l’impression d’une certitude, et d’utiliser un vocabulaire probabiliste et contextualisé. :contentReference[oaicite:14]{index=14}

### 5. Diversité, non-discrimination et équité
ALTAI rappelle qu’un système peut intégrer des biais historiques, des données incomplètes ou non représentatives, et que ces biais peuvent conduire à de la discrimination directe ou indirecte. Il recommande de mettre en place une stratégie pour éviter de créer ou renforcer des biais injustes, et de tester et monitorer ces biais tout au long du cycle de vie. 

Dans un projet d’attrition, c’est un point majeur : certaines variables peuvent être corrélées à des caractéristiques sensibles ou protégées, et un modèle peut “apprendre” des corrélations qui ne doivent pas guider des décisions. La décision d’équipe est donc de traiter l’équité comme un sujet de contrôle, pas comme une hypothèse implicite. Concrètement, nous proposons une évaluation de l’équité par groupes (par exemple en comparant les taux de faux positifs et faux négatifs entre groupes), et un mécanisme clair de remontée d’alerte si un biais ou une performance faible est détecté. 

ALTAI insiste aussi sur la participation des parties prenantes et sur le fait de consulter les personnes potentiellement affectées. Pour un usage RH, cela justifie une recommandation de dialogue interne et, selon le contexte, une discussion avec les représentants du personnel avant une utilisation opérationnelle du score. :contentReference[oaicite:17]{index=17}

### 6. Bien-être sociétal et environnemental
ALTAI invite à regarder l’impact au-delà de la performance technique, en évaluant les effets sociétaux plus larges et les risques d’effets négatifs sur la société ou les institutions. Même si notre cas n’est pas un système “public”, l’idée reste pertinente : un modèle RH peut influencer la culture interne, la confiance, et la relation entre employés et management. :contentReference[oaicite:18]{index=18}

Le bénéfice attendu de notre projet est d’aider à prévenir des départs en agissant plus tôt sur des signaux organisationnels et sur l’expérience employé. Le risque, lui, est de normaliser une logique de contrôle ou d’augmenter l’anxiété liée au fait d’être “scoré”. Notre proposition de garde-fous vise donc à maintenir l’objectif initial : accompagner et améliorer, pas surveiller. À cela s’ajoute une vigilance sur la sobriété : les modèles utilisés restent relativement légers et ne nécessitent pas une infrastructure lourde, ce qui limite l’impact environnemental, tout en restant proportionné à l’usage.

### 7. Responsabilité
ALTAI souligne que la responsabilité implique la possibilité d’audit, une gestion des risques documentée et transparente, et l’existence de mécanismes de recours lorsque des impacts injustes ou défavorables se produisent. Le document insiste aussi sur le fait que des tensions peuvent exister entre exigences, et que les compromis doivent être explicités, argumentés et tracés. 

Dans notre projet, cela se traduit par une gouvernance proposée au client : clarifier qui est responsable de la donnée, du modèle, de l’accès aux scores, et des décisions RH. Nous recommandons aussi une logique d’auditabilité (traces des versions, des données d’entraînement, des évaluations et des accès), ainsi qu’un processus de signalement si un usage pose problème. ALTAI mentionne également la nécessité de mécanismes de redress, ce qui, dans un contexte RH, revient à prévoir la possibilité pour un employé de demander une revue d’une décision ou de signaler un usage inapproprié. 

### Conclusion
En appliquant ALTAI partie par partie, nous avons cherché à faire de l’éthique un élément continu du projet plutôt qu’un ajout final. Les décisions prises en équipe s’articulent autour d’un principe commun : réduire le risque d’atteinte aux personnes en cadrant l’usage (autonomie et supervision), en sécurisant la donnée (confidentialité et gouvernance), en rendant les résultats compréhensibles (transparence), en contrôlant les effets potentiellement injustes (équité), et en mettant en place des mécanismes de suivi, d’audit et de recours (responsabilité). 
