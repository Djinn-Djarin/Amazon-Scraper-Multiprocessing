import gc
import os
import re
import time
import psutil
import asyncio
import random
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from playwright.async_api import Page
from multiprocessing import Process,Semaphore
from playwright.async_api import async_playwright
from typing import List, Literal, Dict, Any
from utils import \
        handle_captcha, \
        handle_network_response, \
        spoof_browser_fingerprint, \
        create_spoofed_context, \
        check_internet, \
        csv_audit_general



LOG_FILE = "scraper_async.log"

# Logging setup
logging.basicConfig(
    filename=LOG_FILE,
    filemode='w',       
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

def log(message):
    logging.info(message)



async def check_status(page: Page, asin: str) -> Literal[
    'Rush Hour', 'Suppressed Warning Found', 'Live', 'Suppressed Asin Changed', 'Suppressed Detail Page Removed', 'Suppressed Default'
]:
    """
    Checks the status of an ASIN on an Amazon product page.

    Args:
        page (Page): The Playwright page object currently loaded.
        asin (str): The ASIN to verify against the page content.

    Returns:
        Literal: A string status indicating one of:
            - 'Rush Hour': If Amazon's rush hour block page is detected.
            - 'Suppressed Warning Found': If a suppression warning is visible.
            - 'Live': If the ASIN page is present and matches the expected ASIN.
            - 'Suppressed Asin Changed': If the loaded ASIN differs from expected.
            - 'Suppressed Detail Page Removed': If no product page is visible after waiting.
            - 'Suppressed Default': Default status if none of the above apply.
    """
    status: str = 'Suppressed Default'
    current_url: str = page.url

    # Expected rush hour message prefix
    expected_message_start = "Oops! It's rush hour and traffic is piling up on that page."

    # Check for rush hour message in a <center> element
    rush_hour_element = await page.query_selector("center")
    if rush_hour_element:
        text_content = await rush_hour_element.text_content()
        clean_text = ' '.join([line.strip() for line in text_content.splitlines() if line.strip()])
        if clean_text.startswith(expected_message_start):
            return 'Rush Hour'

    if 'dp' in current_url:
        # Check if suppression warning is present
        suppressed_element = await page.query_selector(".h1")
        if suppressed_element:
            status = 'Suppressed Warning Found'
        else:
            # Locate the ASIN  element
            asin_element = page.locator('div[data-card-metrics-id^="tell-amazon-desktop_DetailPage_"] div[data-asin]')
            try:
                await asin_element.wait_for(state="visible", timeout=20000)
                main_data_asin_val = await asin_element.get_attribute("data-asin")
                if main_data_asin_val == asin:
                    status = 'Live'
                else:
                    status = 'Suppressed Asin Changed'
            except Exception:
                status = 'Suppressed Detail Page Removed'

    return status



async def scrape_page(
        page:Page, 
        asin:str,
        file_name:str,
        context_settings:Dict[str, Any],
        worker_id:int
        ):
    
    """
        Main script tp Scrapes an Amazon product page for a given ASIN.

        Args:
            page (Page): Playwright page instance.
            asin (str): ASIN of the product.
            file_name (str): File path to save audit CSV data.
            context_settings (dict): Browser context spoofing configs.
            worker_id (int): Worker identifier for logging/audit tracking.

        Returns:
            dict: Dictionary containing product and scrape status information.
    """
    result = {
                        'index':worker_id+1,
                        "asin":  asin,
                        "status": 'Suppressed',
                        "brand_name":"N/A",
                        "browse_node":  "N/A",
                        "title":  "N/A",
                        "reviews":  "0",
                        "ratings":  "0",
                        "variations":"N/A",
                        "deal":  "N/A",
                        "seller": "N/A",
                        "image_len":  0,
                        "video":"N/A",
                        "main_img_url": "N/A",
                        "bullet_point_len": 0,
                        "bestSellerRank": f"",
                        "price":  "N/A",
                        "MRP":  0,
                        "availability": "N/A",
                        "description":  "N/A",
                        "A_plus":"N/A",
                        "store_link": "N/A",
                    }
    try:
        page.on("response", handle_network_response)
        await spoof_browser_fingerprint(page, context_settings)
        await page.goto(
            f"https://www.amazon.in/dp/{asin}", timeout=20000, wait_until="domcontentloaded"
        )
        await page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(random.uniform(3, 5))

    except Exception as e:
        logging.warning(f"Error for ASIN {asin}: {e}")
        result['status'] = 'Suppressed Page Timeout'
        await csv_audit_general(result, file_name)
        return result


    
    # CAPTCHA Handling
    captcha_result = await handle_captcha(page)
    if not captcha_result:
        logging.info(f"captcha not solved for {asin}")
        
        result['status'] = 'Suppressed Captcha Failure'
        await csv_audit_general(result, file_name)
        return result
    

    # Look for 'Continue Shopping' button
    button = page.locator("button", has_text=re.compile(r'continue shopping', re.IGNORECASE))
    if await button.count() > 0 and await button.is_visible():
        logging.info('Continue Shopping button found')
        try:
            await button.click()
        except Exception as e:
            logging.error(f"Button click failed: {e}")
            result['status'] = 'Suppressed Continue Button'
            await csv_audit_general(result, file_name)
            return result
        
    status = await check_status(page, asin)
    if status in ['Suppressed Captcha Failure', 'Suppressed Img Section Not Found', 'Suppressed Warning Found', 'Rush Hour']:
        result['status'] = status
        await csv_audit_general(result, file_name)
        return result

    try:
        # SUPPRESSION STATUS 
        brand_name_element = page.locator("a#bylineInfo").first
        if await brand_name_element.count() > 0:
            await brand_name_element.wait_for(state="visible", timeout=10000)
            raw_brand = (await brand_name_element.text_content()).strip()
            brand_name = re.sub(r'^(Visit the\s+)?(.*?)(\s+Store)?$', r'\2', raw_brand).strip()
        else:
            logging.info("brand name link not found")
            brand_name = 'N/A'

        # availability
        availability_locator = page.locator("div#availability").first
        if await availability_locator.count() > 0:
            availability_text = await availability_locator.inner_text()
            availability = availability_text.strip().split()[0] if availability_text else None
        else:
            availability = None

        # browse node
        # Locate all <a> elements inside the breadcrumb container
        breadcrumb_locator = page.locator(
            "div#wayfinding-breadcrumbs_feature_div ul.a-unordered-list.a-horizontal.a-size-small a"
        )

        # Check if any breadcrumb links exist
        if await breadcrumb_locator.count() > 0:
            # Get text content of all <a> elements
            browse_node_list = await breadcrumb_locator.all_inner_texts()
            # Join them with ' > '
            browse_node = " > ".join(text.strip() for text in browse_node_list)
        else:
            browse_node = None


        # title
        title_locator = page.locator("span#productTitle").first
        try:
            await title_locator.wait_for(state="visible", timeout=5000)  # timeout in milliseconds
            title = await title_locator.text_content()
            if title:
                title = title.strip()
        except Exception as e:
            # logging.error(f"Title not found : {e}")
            title = ''


        # reviews
        reviews_locator = page.locator("span#acrPopover").first
        try:
            await reviews_locator.wait_for(state="visible", timeout=10000)
            reviews_title = await reviews_locator.get_attribute("title")
            reviews = reviews_title.split(" ")[0].strip() if reviews_title else "0"
        except:
            reviews = "0"
            logging.error(f"review {reviews}")

        # ratings
        ratings_locator = page.locator("span#acrCustomerReviewText").first
        if await ratings_locator.count() > 0:
            ratings_text = await ratings_locator.text_content()
            ratings = ratings_text.split(" ")[0].replace(",", "").strip() if ratings_text else "0"
        else:
            ratings = "0"

        # variations
        variations_locator = page.locator(
            "#twister-plus-inline-twister, "
            "#variation_color_name, "
            "#variation_size_name, "
            "#inline-twister-row-pattern_name, "
            "#variation_style_name"
        )
        variations = "Available" if await variations_locator.count() > 0 else "NA"

        # deal
        deal_locator = page.locator("span.dealBadgeTextColor").first
        deal = "NA"
        if await deal_locator.count() > 0:
            deal_text = await deal_locator.text_content()
            if deal_text and "Limited time deal" in deal_text:
                deal = "Available"

        # seller
        seller_locator = page.locator("#sellerProfileTriggerId").first
        seller = (await seller_locator.text_content()).strip() if await seller_locator.count() > 0 else None

        # images
        image_locators = page.locator("#altImages img")
        image_count = await image_locators.count()
        img_urls = [await image_locators.nth(i).get_attribute("src") for i in range(image_count)]


        # video
        video_locator = page.locator("li.videoThumbnail img").first
        video = "Available" if await video_locator.count() > 0 else "Not Available"

        # main image url
        main_img_url = None
        try:
            ul_locator = page.locator(
                "ul.a-unordered-list.a-nostyle.a-button-list.a-vertical.a-spacing-top-micro.gridAltImageViewLayoutIn1x7"
            ).first
            if await ul_locator.count() == 0:
                ul_locator = page.locator(
                    "ul.a-unordered-list.a-nostyle.a-button-list.a-vertical.a-spacing-top-extra-large.regularAltImageViewLayout"
                ).first

            if await ul_locator.count() > 0:
                img_locators = ul_locator.locator("img")
                count = await img_locators.count()
                for i in range(count):
                    src = await img_locators.nth(i).get_attribute("src")
                    if src and src.endswith(".jpg"):
                        main_img_url = src.replace("SS100", "SS500")
                        break
        except Exception as e:
            logging.warning(f"Error fetching main image URL: {e}")

        # bullet points length
        bullet_point_len = 0
        try:
            ul_locator = page.locator("div#feature-bullets ul.a-unordered-list.a-vertical.a-spacing-mini").first
            if await ul_locator.count() > 0:
                bullet_point_len = await ul_locator.locator("li").count()
        except Exception as e:
            logging.warning(f"Error counting bullet points: {e}")

        # best sellers rank
        bsr1, bsr2 = "Not Available", "Not Available"
        try:
            table = page.locator("table#productDetails_detailBullets_sections1").first
            if await table.count() > 0:
                th_elements = await table.locator("th").all()
                best_sellers_th = None
                for th in th_elements:
                    text = (await th.text_content() or "").strip()
                    if text == "Best Sellers Rank":
                        best_sellers_th = th
                        break
                if best_sellers_th:
                    best_sellers_td = await best_sellers_th.evaluate_handle("th => th.nextElementSibling")
                    if best_sellers_td:
                        span_texts = await best_sellers_td.eval_on_selector_all(
                            "span", "spans => spans.map(s => s.textContent.trim())"
                        )
                        ranks_str = " ".join(span_texts)
                        ranks = ranks_str.split("#")[1:3]
                        if span_texts:
                            ranks = ranks_str.split("#")[1:3]
                            if len(ranks) < 2:
                                ranks += ["Not Available"] * (2 - len(ranks))
                            bsr1, bsr2 = ranks[0], ranks[1]
                        else:
                            bsr1, bsr2 = "Not Available", "Not Available"

        except Exception as e:
                            logging.warning(f"Error fetching Best Sellers Rank: {e}")

        # price and MRP
        price = None
        mrp = 0

        price_loc = page.locator("span.a-price-whole")
        if await price_loc.count() > 0:
            price_text = await price_loc.first.text_content()
            if price_text:
                price = price_text.strip()

        mrp_label = None
        spans = page.locator("span").first
        spans_count = await spans.count()
        for i in range(spans_count):
            text = await spans.nth(i).text_content()
            if text and "M.R.P." in text:
                mrp_label = spans.nth(i)
                break

        if mrp_label:
            price_element = mrp_label.locator(".a-price .a-offscreen").first
            if await price_element.count() > 0:
                mrp_text = await price_element.first.text_content()
                if mrp_text:
                    mrp_text = mrp_text.replace('₹', '').replace(',', '').strip()
                    try:
                        mrp = float(mrp_text)
                    except ValueError:
                        mrp = 0

        # description
        description = "Not Available"
        desc_loc = page.locator("#productDescription").first
        if await desc_loc.count() > 0:
            desc_text = await desc_loc.first.text_content()
            if desc_text:
                description = desc_text.strip()

        # A+ content
        aplus_data = page.locator("#aplus")
        A_plus = "Available" if await aplus_data.count() > 0 else "NA"

        # store link
        store_link = ""
        byline_info = page.locator("a#bylineInfo").first
        if await byline_info.count() > 0:
            href = await byline_info.first.get_attribute("href")
            if href:
                store_link = f"http://amazon.in{href}"

        result = {
            'index':worker_id+1,
            "asin": asin ,
            "status": status,
            "brand_name": brand_name or "N/A",
            "browse_node": browse_node or "N/A",
            "title": title or "N/A",
            "reviews": reviews or "0",
            "ratings": ratings or "0",
            "variations": variations or "N/A",
            "deal": deal or "N/A",
            "seller": seller or "N/A",
            "image_len": image_count or 0,
            "video": video or "N/A",
            "main_img_url": main_img_url or "N/A",
            "bullet_point_len": bullet_point_len or 0,
            "bestSellerRank": f"{bsr1}, {bsr2}",
            "price": price or "N/A",
            "MRP": mrp or 0,
            "availability": availability or "N/A",
            "description": description or "N/A",
            "A_plus": A_plus or "N/A",
            "store_link": store_link or "N/A",
        }
        try:
            await csv_audit_general(result, file_name)
        except Exception as e:
            logging.info(f"error at the end of csv file {e}")
        await page.close()
        gc.collect()
        return result
    except Exception as e:
        logging.info(f" error {e}")
        result['status'] = 'Unknown'
        await csv_audit_general(result, file_name)
        await page.close()
        gc.collect()
        return result
    





def browser_worker_with_list(
    url_list: List[str],
    worker_id: int,
    max_tabs: int,
    file_name: str,
    headless: bool
) -> str:
    """
    Asynchronously launches a Playwright browser to scrape a list of ASIN URLs.

    Args:
        url_list (List[str]): List of ASIN URLs to scrape.
        worker_id (int): Unique identifier for this worker instance.
        max_tabs (int): Number of concurrent pages (tabs) allowed per browser instance.
        file_name (str): Path to the output CSV file for saving results.
        headless (bool): If True, runs the browser in headless mode (no UI).

    Returns:
        str: Confirmation message upon task completion.
    """

    async def browser_worker_async() -> str:
        """
        Inner asynchronous function to run the browser context,
        manage the scraping queue, and handle multiple concurrent tabs.
        
        Returns:
            str: Confirmation message upon async task completion.
        """
        logging.info(f"[Browser {worker_id}] Starting browser for {len(url_list)} ASINs")

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=headless,
                args=[
                    "--disable-quic",
                    "--disable-http2",
                    "--disable-blink-features=AutomationControlled",
                    "--enable-webgl",
                    "--use-gl=swiftshader",
                    "--enable-accelerated-2d-canvas",
                    "--disable-features=UseDnsHttpsSvcbAlpn",
                ]
            )

            logging.info(f"[Browser {worker_id}] Browser opened")

            context, context_settings = await create_spoofed_context(browser)

            queue: asyncio.Queue[str] = asyncio.Queue()
            for url in url_list:
                await queue.put(url)

            logging.info(f"[Browser {worker_id}] Queue size: {queue.qsize()}")

            total_asins: int = len(url_list)
            progress_bar = tqdm(total=total_asins, desc=f"Worker {worker_id}", position=worker_id)
            processed_asins: int = 0

            async def worker() -> None:
                nonlocal processed_asins
                while True:
                    try:
                        url = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    if not check_internet():
                        logging.error(f"[Browser {worker_id}] No Internet connection. Waiting to retry {url}...")
                        await asyncio.sleep(5)

                    while psutil.cpu_percent(interval=1) > 80:
                        logging.warning(f"[Browser {worker_id}] High CPU usage detected. Waiting...")
                        await asyncio.sleep(2)

                    try:
                        page = await context.new_page()
                        logging.info(f"[Browser {worker_id}] Scraping {url}")
                        await scrape_page(page, url, file_name, context_settings, worker_id)
                        await asyncio.sleep(1)
                        await page.close()
                        gc.collect()
                    except Exception as e:
                        logging.error(f"[Browser {worker_id}] Error scraping {url}: {e}")
                    finally:
                        processed_asins += 1
                        progress_bar.update(1)

            # Create up to `max_tabs` worker tasks
            tasks = [asyncio.create_task(worker()) for _ in range(max_tabs)]

            # Wait for all workers to complete
            await asyncio.gather(*tasks, return_exceptions=True)

            progress_bar.close()
            await browser.close()
            logging.info(f"🔴 [Browser {worker_id}] Browser closed")
            gc.collect()
            logging.info(f"[Browser {worker_id}] ✅ Finished.")
            return "task completed"

    asyncio.run(browser_worker_async())
    return "task completed"



def start_browser_process(
    url_list: List[str],
    worker_id: int,
    max_tabs: int,
    file_name: str,
    sem: Semaphore,
    headless: bool
) -> str:
    """
    Starts a browser scraping process for a given list of ASIN URLs.

    Args:
        url_list (List[str]): List of ASIN URLs to scrape.
        worker_id (int): Unique identifier for the worker/process.
        max_tabs (int): Maximum number of tabs allowed per browser instance.
        file_name (str): Path to the CSV file where results will be saved.
        sem (multiprocessing.Semaphore): Semaphore to control concurrent process count.
        headless (bool): If True, runs the browser in headless (no-UI) mode.

    Returns:
        str: Confirmation message upon task completion.
    """
    try:
        browser_worker_with_list(url_list, worker_id, max_tabs, file_name, headless)
    finally:
        sem.release()

    return "task completed"


def main() -> None:

    """
    Main function to coordinate the scraping process.

    - Parses command-line arguments.
    - Validates the input Excel file.
    - Reads and deduplicates ASINs.
    - Prepares output directory and file.
    - Divides ASINs into batches based on browser size.
    - Launches concurrent scraping processes while respecting the max browser limit.
    - Waits for all processes to complete.
    """
     
    parser = argparse.ArgumentParser(description="Amazon scraping script")
    parser.add_argument('--input', type=str, required=True, help='Excel file only')
    args = parser.parse_args()

    print(args.input, 'args')
    if not args.input.lower().endswith('.xlsx'):
        raise ValueError ("enter xlsx file only")
    
      # Configuration values
    browser_size: int = 1       # Max ASINs per browser instance
    max_tabs: int = 1           # Max tabs per browser
    max_browser: int = 50       # Max concurrent browser processes
    headless: bool = True       # Run browsers in headless mode
    
    # Prepare output file name and directory
    input_filename: str = os.path.splitext(os.path.basename(args.input))[0]
    timestamp: str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    output_dir: str = "./output"
    os.makedirs(output_dir, exist_ok=True)
    output_file: str = os.path.join(output_dir, f"{input_filename}_{timestamp}.csv")

    # Read and preprocess ASIN data
    data: pd.DataFrame = pd.read_excel(args.input)
    df_unique: pd.DataFrame = data.drop_duplicates(subset=['asins'])
    urls: List[str] = df_unique["asins"].dropna().astype(str).to_list()


    # Split ASINs into batches based on browser_size
    asin_batches: List[List[str]] = [
        urls[i:i + browser_size] for i in range(0, len(urls), browser_size)
    ]

    total_batches: int = len(asin_batches)
    
    logging.info(f"📊 Total Browsers (batches) to be launched: {total_batches}")
    logging.info(f"📄 Output file: {output_file}")

     # Semaphore to control max concurrent processes
    sem = Semaphore(max_browser)
    active_processes: List[Process] = []
 
    for batch_id, batch_urls in enumerate(asin_batches):
        sem.acquire()
        p: Process = Process(target=start_browser_process, args=(batch_urls, batch_id, max_tabs,  output_file, sem,headless))
        p.start()
        active_processes.append(p)

        # Keep only alive processes in the list
        active_processes = [proc for proc in active_processes if proc.is_alive()]

        while len(active_processes) >= max_browser:
            time.sleep(1)
            active_processes = [proc for proc in active_processes if proc.is_alive()]
        time.sleep(0.5)

    # Wait for all remaining processes to finish
    for p in active_processes:
        p.join()

    logging.info("✅All ASIN batches processed.")

if __name__ == "__main__":
    main()