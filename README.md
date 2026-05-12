# Hidden Mascot Detector

Een AI-project voor school waarin een computer-vision model een zelfgemaakte mascotte zoekt in drukke afbeeldingen. De gebruiker uploadt een afbeelding, het model detecteert de mascotte intern, en de gebruiker klikt op de afbeelding om te raden waar de mascotte zit.

Het project is begonnen als een Where Is Wally/Waldo idee, maar is aangepast naar een eigen mascotte. Daardoor is de dataset beter controleerbaar, is er geen afhankelijkheid van zeldzame of auteursrechtelijk gevoelige Wally-afbeeldingen, en blijft de AI-opdracht duidelijk: zelf data maken, zelf trainen, en een interactieve toepassing bouwen.

**Waarom dit project?**
- **Eigen dataset:** de trainingsbeelden worden synthetisch gegenereerd met perfecte labels.
- **Echte AI-pipeline:** dataset maken, YOLO trainen, model gebruiken in een webapp.
- **Interactief:** de gebruiker kan de mascotte downloaden, in een eigen afbeelding verwerken, uploaden, en daarna de voorspelde bounding box en confidence bekijken.
- **Uitbreidbaar:** later kan de synthetische data gemengd worden met eigen CVAT-labels.

**MVP**
- Genereer een dataset met een verborgen mascotte.
- Train een YOLOv8 object-detection model op de klasse `mascot`.
- Download de mascotte-PNG vanuit de webapp.
- Upload een eigen bewerkte afbeelding waarin de mascotte is verstopt.
- Detecteer de mascotte met het getrainde model.
- Teken de voorspelde bounding box en toon de confidence.
- Laat de gebruiker klikken en geef tekst-feedback.

**Hoe het werkt**
1. `tools/generate_synthetic_mascot_dataset.py` maakt drukke zoekplaat-achtige afbeeldingen.
2. Bij elke afbeelding wordt automatisch een YOLO-label geschreven rond de mascotte.
3. YOLOv8 traint op deze dataset.
4. De FastAPI-backend laadt het model via `MODEL_PATH`.
5. De frontend tekent de voorspelde bounding box op de canvas en toont de confidence score.
6. De frontend stuurt klik-coördinaten naar de backend.
7. De backend vergelijkt de klik met het midden van de gedetecteerde bounding box.

**Projectstructuur**
- `app/` — FastAPI-backend voor detectie en klik-feedback.
- `frontend/` — eenvoudige HTML/JS webinterface.
- `tools/` — scripts om datasets voor te bereiden of te genereren.
- `models/` — opgeslagen modelgewichten, niet in git.
- `data/` — datasets, niet in git.
- `runs/` — YOLO-trainingsoutput, niet in git.

## Installatie

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Synthetische dataset maken

Maak een dataset met automatisch gelabelde afbeeldingen:

```bash
python3 tools/generate_synthetic_mascot_dataset.py --output data/synthetic-mascot --train 240 --val 60 --difficulty hard
```

De generator ondersteunt `easy`, `medium` en `hard`. Voor het project is `hard` het interessantst, omdat de afbeeldingen meer drukte, lookalikes en gedeeltelijke overlap bevatten. De synthetische pipeline past ook sterke augmentaties toe, waaronder blur, brightness/contrast-variatie, rotatie, scaling, perspective transforms en beeldruis.

Je kunt ook je **eigen mascotte** gebruiken. Maak daarvoor een afbeelding met transparante achtergrond, bijvoorbeeld:

```text
assets/mascot.png
```

Gebruik die mascotte dan zo:

```bash
python3 tools/generate_synthetic_mascot_dataset.py --output data/my-mascot --train 500 --val 100 --difficulty hard --mascot assets/mascot.png
```

Als je eigen achtergrondfoto's hebt, zet die bijvoorbeeld in:

```text
assets/backgrounds/
```

Dan kun je de mascotte automatisch op jouw achtergronden laten plaatsen:

```bash
python3 tools/generate_synthetic_mascot_dataset.py --output data/my-mascot --train 500 --val 100 --difficulty hard --mascot assets/mascot.png --backgrounds assets/backgrounds
```

Dat is de handigste manier om zelf data te maken zonder alles handmatig in CVAT te labelen. Het script plakt de mascotte op de afbeelding en schrijft automatisch de juiste bounding box.

De dataset krijgt deze YOLO/Ultralytics-structuur:

```text
data/synthetic-mascot/
  data.yaml
  images/train/
  images/val/
  labels/train/
  labels/val/
```

## Model trainen

Train YOLOv8 op de gegenereerde dataset:

```bash
yolo detect train data=data/synthetic-mascot/data.yaml model=yolov8n.pt epochs=30 imgsz=640
```

COCO is hiervoor vooral nuttig als pretrained startpunt via bijvoorbeeld `yolov8n.pt`, of als bron van extra achtergrondfoto's voor `--backgrounds`. Gebruik COCO niet als hoofddataset voor deze taak: COCO bevat geen label voor jouw specifieke mascotte, terwijl de synthetische dataset precies die klasse met correcte bounding boxes genereert.

Na training staat het beste model meestal hier:

```text
runs/detect/train/weights/best.pt
```

Gebruik het getrainde model in de app:

```bash
MODEL_PATH=runs/detect/train/weights/best.pt TARGET_CLASS=mascot uvicorn app.main:app --reload
```

Open daarna `frontend/index.html` in de browser. Je kunt de mascotte-PNG downloaden, zelf in een afbeelding verwerken, en die afbeelding uploaden. De frontend tekent de voorspelde bounding box en toont de confidence score.

Je kunt ook een testafbeelding uploaden uit bijvoorbeeld:

```text
data/synthetic-mascot/images/val/
```

## Eigen CVAT-labels gebruiken

Als je later echte afbeeldingen of zelfgemaakte zoekplaten wil labelen in CVAT, kan dat nog steeds. Maak in CVAT een label met de naam `mascot`, exporteer als YOLO, pak de zip uit en zet die om:

```bash
python3 tools/prepare_cvat_yolo.py data/cvat-export --output data/cvat-mascot --fallback-class mascot
```

Daarna kun je op die dataset trainen:

```bash
yolo detect train data=data/cvat-mascot/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

Je kunt ook synthetische data gebruiken als snelle start en daarna verbeteren met CVAT-data. Dat is een sterk verhaal voor het project: eerst gecontroleerde data genereren, daarna testen of het model ook werkt op realistischer beeldmateriaal.

## Feedbacklogica

De backend berekent de afstand tussen de klik en het midden van de gedetecteerde box:

- `Gevonden`: klik zit dicht bij het midden van de box.
- `Dichtbij`: klik zit in de buurt.
- `Ver weg`: klik zit duidelijk naast de mascotte.
- `Geen detectie`: het model vond geen mascotte.
