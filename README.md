# Where Is Wally — Projectoverzicht

Een persoonlijk projectidee voor school: bouw een interactieve webapp waarin een computer-vision model zoekt naar Wally op een zoekplaat. De gebruiker uploadt een afbeelding, het model detecteert Wally, en de gebruiker kan vervolgens op de afbeelding klikken om te proberen Wally te vinden.

**Waarom dit project?**
- **Eigen idee:** een persoonlijke en creatieve toepassing.
- **Realistisch:** uitvoerbaar met een klein model en een beperkte dataset.
- **Interactief:** klik-feedback maakt het spel aantrekkelijk voor gebruikers.

**MVP (Minimal Viable Product)**
- Upload één zoekplaat.
+- Detecteer Wally met een object-detection model.
-- De bounding box wordt alleen intern gebruikt door het systeem en wordt niet aan de gebruiker getoond.
-- De gebruiker blijft de afbeelding zien zonder box en kan blijven klikken om Wally te vinden.
-- Geef per klik feedback: `Gevonden`, `Dichtbij` of `Ver weg`.

**Hoe het werkt (kort)**
1. Gebruiker uploadt een afbeeldingsbestand.
2. Backend voert object-detection uit en houdt één of meerdere bounding boxes intern bij.
3. Frontend toont alleen de afbeelding (de box wordt niet getoond aan de gebruiker).
4. Gebruiker klikt op de afbeelding; frontend stuurt klik-coördinaten naar de backend.
5. Backend berekent de afstand tussen de klik en het midden van de (interne) Wally-box en retourneert alleen een feedback-label op basis van vooraf ingestelde drempels.

**Technische opzet (suggestie)**
- Model: klein(er) object detection model (bijv. YOLOv5/YOLOv8 of een Tiny-variant van Detectron/SSD) voor snelle inference. De gegenereerde bounding box blijft intern en wordt niet naar de frontend gestuurd, zodat gebruikers kunnen blijven klikken en zelf Wally zoeken.
- Dataset: handmatig gelabelde zoekplaten met een box rond Wally; start met 50–200 afbeeldingen en augmentatie.
- Backend: Python + Flask/FastAPI voor inference endpoints.
- Frontend: eenvoudige HTML/JS (bijv. React of plain JS) die afbeeldingen toont en klik-coördinaten stuurt.

**Afstand- en feedbacklogica**
- Bereken Euclidische afstand tussen klik (x,y) en box-centroid.
- Stel drempels in (voorbeeld):
	- afstand <= 0.2 * diag(box) → `Gevonden`
	- afstand <= 0.5 * diag(box) → `Dichtbij`
	- anders → `Ver weg`

**Installatie & snelstart (voorbeeld)**
1. Maak een virtuele omgeving en installeer dependencies:

```bash
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
```

2. Start de backend:

```bash
uvicorn app.main:app --reload
```

3. Open de frontend (lokale HTML of via een dev-server) en upload een afbeelding.

**Uitbreidingen voor later**
- Meerdere klikpogingen en scoring (spelmodus).
- Detectie van meerdere personen en onderscheid tussen echte Wally's en lookalikes.
- Model fine-tuning op grotere datasets voor robuustere detectie.

**Bestanden & organisatie (aanbevolen)**
- `app/` — backend code (inference, API).
- `frontend/` — webinterface.
- `models/` — opgeslagen modelgewichten.
- `data/` — gelabelde zoekplaten en annotaties.
- `requirements.txt` — Python dependencies.

**Bijdragen & contact**
Feedback, ideeën of hulp zijn welkom — open een issue of stuur een bericht.

