<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%2B-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/playwright-async-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/captcha-solving-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/fingerprint-spoofing-purple?style=for-the-badge" />
</p>

# 📦 Amazon Scraper

### With Playwright Async, Captcha Handling & Fingerprint Spoofing

---

## 📑 Features

- ✅ Scrapes up to **6000 products/hour**
- ✅ Asynchronous scraping via **Playwright**
- ✅ Multiprocessing for throttling and **true parallel processing**
- ✅ **Captcha detection** and auto-solving via `amazon-captcha`
- ✅ **Browser fingerprint spoofing** (user agent, timezone, geolocation, hardware concurrency, etc.)
- ✅ **Internet connection checks** before each scrape
- ✅ Optional **CPU usage limits**
- ✅ Real-time progress tracking via `tqdm`
- ✅ Graceful error handling and logging

---

## 📄 Data Fields Scraped

For each product page, this scraper extracts:

- **ASIN**
- **Brand Name**
- **Status**
- **Product Title**
- **Price**
- **MRP**
- **Rating**
- **Number of Reviews**
- **Browse Node**
- **Availability Status**
- **Product Description**
- **Bullet Point Features**
- **Seller Name**
- **Image URLs**
- **Product URL**
- **Store Link**

> 📌 *You can easily adjust fields inside your `scrape_page()` function.*

---

## ⚙️ Performance Tuning (Important)

**Configuration values:**

```python
browser_size: int = 1       # Max ASINs per browser instance
max_tabs: int = 1           # Max tabs per browser
max_browser: int = 50       # Max concurrent browser processes
headless: bool = True       # Run browsers in headless mode
```

**Notes:**

- More browsers → more RAM → lower detection
- More tabs → less RAM → higher throughput → higher detection risk  
*"Detection" means captchas to solve*

---

## 🛠️ Requirements

- Python 3.12+

**Install dependencies:**

```bash
pip install -r requirements.txt
playwright install
```

---

## 🚀 Run the Scraper

```bash
python main.py --input './your_file.xlsx'
```

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 💬 Feedback

⭐ Star the repo or open an issue if this helped you!
