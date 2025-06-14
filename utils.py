import os
import csv
import json
import random
import socket
import aiohttp
import aiofiles
from datetime import datetime
from PIL import Image
from io import BytesIO
from typing import Dict, Any, Tuple
from amazoncaptcha import AmazonCaptcha
from playwright.async_api import Response, Page, Browser, BrowserContext


def check_internet() -> bool:
    """
    Check if the machine has an active internet connection by attempting 
    to connect to Google's public DNS server (8.8.8.8) on port 53.

    Returns:
        bool: True if the connection is successful, False otherwise.
    """
    try:
        socket.setdefaulttimeout(5)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(("8.8.8.8", 53))
        return True
    except (socket.timeout, socket.gaierror, socket.error):
        return False


async def handle_network_response(response: Response) -> None:
    """
    Handle network responses during page navigation. 
    If the response URL contains 'dp' and status is 200, 
    attempts to parse the response body as JSON.

    Args:
        response (Response): The network response object from Playwright.
    """
    try:
        if "dp" in response.url and response.status == 200:
            body = await response.body()

            try:
                json_data = json.loads(body)
                # You can optionally do something with json_data here
            except json.JSONDecodeError:
                pass
    except Exception as e:
        # Optionally log the error if needed
        pass




async def is_captcha_present(page: Page) -> bool:
    """
    Checks whether a captcha input field is present on the given page.

    Args:
        page (Page): The Playwright page instance to inspect.

    Returns:
        bool: True if a captcha input element is found, False otherwise.
    """
    try:
        captcha_input = await page.query_selector("input#captchacharacters")
        captcha_present = captcha_input is not None
        return captcha_present
    except Exception as e:
        print(f"Error checking captcha presence: {e}")
        return False



async def solve_captcha(page: Page) -> bool:
    """
    Attempts to detect and solve an Amazon captcha on the given page.

    This function locates the captcha image, downloads it, solves it using
    the AmazonCaptcha library, and submits the solution. It waits for the
    page to fully load afterward.

    Args:
        page (Page): The Playwright page instance to work with.

    Returns:
        bool: True if the captcha was successfully solved and submitted,
              False otherwise.
    """
    try:
        captcha_image = await page.query_selector("img[src*='captcha']")
        if not captcha_image:
            print("No captcha image found.")
            return False

        captcha_src = await captcha_image.get_attribute("src")
        print(f"Captcha source URL: {captcha_src}")

        async with aiohttp.ClientSession() as session:
            async with session.get(captcha_src) as response:
                if response.status == 200:
                    content = await response.read()
                    img = Image.open(BytesIO(content))

                    captcha = AmazonCaptcha.fromlink(captcha_src)
                    solution = captcha.solve()
                    print(f"Captcha solution: {solution}")

                    captcha_input = await page.query_selector("#captchacharacters")
                    await captcha_input.click()
                    await captcha_input.type(solution, delay=100)

                    submit_button = await page.query_selector('button[type="submit"]')
                    await submit_button.click()

                    await page.wait_for_load_state("load", timeout=60000)
                    return True

        print("Failed to fetch captcha image.")
        return False

    except Exception as e:
        print(f"Failed to solve captcha: {e}")
        return False


async def handle_captcha(page: Page) -> bool:
    """
    Detects and attempts to solve a captcha if present on the given page.

    This function checks for the presence of a captcha form. If found, it
    attempts to solve it using `solve_captcha`. If successful, it reloads
    the page. It returns True if no captcha is present or if solved successfully,
    and False if solving fails or an exception occurs.

    Args:
        page (Page): The Playwright page instance to check and act upon.

    Returns:
        bool: True if no captcha is present or solved successfully, 
              False otherwise.
    """
    try:
        captcha_present = await is_captcha_present(page)
        if not captcha_present:
            return True

        print("Captcha detected, attempting to solve...")
        captcha_solved = await solve_captcha(page)
        if captcha_solved:
            await page.reload()
            return True

        print("Failed to solve captcha.")
        return False

    except Exception as e:
        print(f"Error while handling captcha: {e}")
        return False

    

async def spoof_browser_fingerprint(page: Page, context_settings: Dict[str, Any]) -> None:

    """
    Spoofs various browser fingerprint attributes via JavaScript injection on the given page.

    This function alters properties such as user agent, platform, languages, hardware concurrency, 
    and device memory to simulate a realistic and randomized browser environment, reducing 
    the likelihood of bot detection.

    Args:
        page (Page): The Playwright page instance to apply fingerprint spoofing to.
        context_settings (Dict[str, Any]): Dictionary containing browser context settings, 
                                           including the 'user_agent' key.

    Returns:
        None
    """
      
    ua = context_settings['user_agent']
    if "Windows" in ua:
        platform = "Win32"
    elif "Macintosh" in ua:
        platform = "MacIntel"
    else:
        platform = "Linux x86_64"

    languages = random.sample([['en-US', 'en'], ['fr-FR', 'fr'], ['de-DE', 'de'], ['es-ES', 'es']], 1)[0]
    plugins = random.sample(range(1, 6), random.randint(1, 5))
    hardware_concurrency = random.choice([2, 4, 8, 16])
    device_memory = random.choice([2, 4, 8, 16])
    brands = [{"brand": "Not-A.Brand", "version": str(random.randint(90, 120))}]
    mobile = random.choice([True, False])

    init_script = f"""
        Object.defineProperty(navigator, 'webdriver', {{
            get: () => false
        }});

        Object.defineProperty(navigator, 'languages', {{
            get: () => {json.dumps(languages)}
        }});

        Object.defineProperty(navigator, 'platform', {{
            get: () => '{platform}'
        }});

        Object.defineProperty(navigator, 'plugins', {{
            get: () => {json.dumps(plugins)}
        }});

        Object.defineProperty(navigator, 'hardwareConcurrency', {{
            get: () => {hardware_concurrency}
        }});

        Object.defineProperty(navigator, 'deviceMemory', {{
            get: () => {device_memory}
        }});

        if (navigator.userAgentData) {{
            Object.defineProperty(navigator, 'userAgentData', {{
                value: {{
                    brands: {json.dumps(brands)},
                    mobile: {str(mobile).lower()},
                    platform: '{platform}'
                }}
            }});
        }}

        Object.defineProperty(window, 'chrome', {{
            value: {{
                runtime: {{}}
            }}
        }});

        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications' ?
                Promise.resolve({{ state: Notification.permission }}) :
                originalQuery(parameters)
        );

        Object.defineProperty(navigator, 'mediaDevices', {{
            value: {{
                enumerateDevices: () => Promise.resolve([])
            }}
        }});
    """

    await page.add_init_script(init_script)


async def create_spoofed_context(browser: Browser) -> Tuple[BrowserContext, Dict[str, Any]]:
    """
    Creates a new browser context with spoofed settings for user agent, viewport, timezone, 
    locale, and geolocation.

    Args:
        browser (Browser): The Playwright browser instance to create the context from.

    Returns:
        Tuple[BrowserContext, Dict[str, Any]]: A tuple containing the created browser context and 
        a dictionary with the spoofed context settings.
    """

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ]

    viewports = [
        {"width": 1366, "height": 768},
        {"width": 1920, "height": 1080},
        {"width": 1440, "height": 900}
    ]

    timezones = [
        "America/New_York",
        "Europe/London",
        "Asia/Kolkata",
        "Australia/Sydney"
    ]

    locales = [
        "en-US",
        "en-GB",
        "fr-FR",
        "de-DE"
    ]

    geolocations = [
        {"longitude": -73.935242, "latitude": 40.730610},
        {"longitude": -0.127758, "latitude": 51.507351},
        {"longitude": 77.594566, "latitude": 12.971599},
        {"longitude": 151.209900, "latitude": -33.865143}
    ]

    # Now pick one of each randomly
    user_agent = random.choice(user_agents)
    viewport = random.choice(viewports)
    timezone = random.choice(timezones)
    locale = random.choice(locales)
    geolocation = random.choice(geolocations)

    context = await browser.new_context(
        user_agent=user_agent,
        viewport=viewport,
        timezone_id=timezone,
        locale=locale,
        geolocation=geolocation,
        permissions=["geolocation"],
        bypass_csp=True
    )

    return context, {
        "user_agent": user_agent,
        "viewport": viewport,
        "timezone": timezone,
        "locale": locale,
        "geolocation": geolocation
    }




async def csv_audit_general(data: Dict[str, str], filepath: str) -> None:
    """
    Asynchronously appends product audit data to a CSV file.
    Creates the directory if it doesn't exist and writes headers if the file is new.

    Args:
        data (Dict[str, str]): A dictionary containing the product audit information.
        filepath (str): The file path to the CSV file where data should be appended.

    Returns:
        None
    """
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    headers = [
        'index',
        "asin",
        "brand_name",
        "status",
        "availability",
        "reviews",
        "ratings",
        "variations",
        "deal",
        "seller",
        "image_len",
        "video",
        "main_img_url",
        "bullet_point_len",
        "bestSellerRank",
        "price",
        "MRP",
        "browse_node",
        "title",
        "description",
        "A_plus",
        "store_link",
        "timestamp",
    ]

    file_exists = os.path.isfile(filepath) and os.stat(filepath).st_size != 0
    data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    async with aiofiles.open(filepath, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)

        if not file_exists:
            await file.write(",".join(headers) + "\n")

        await file.write(",".join([str(data.get(h, "")) for h in headers]) + "\n")
        await file.flush()


    # print(f" Data appended to {filepath}")

