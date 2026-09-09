import json
import logging
import os
import sys
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PORTAL_URL = "https://www.india.gov.in/my-government/schemes"
DEFAULT_DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_DOCS_DIR, "schema")


def scrape_schemes(
    output_base_path: str = DEFAULT_OUTPUT_PATH,
    headless: bool = False,
    slow_mo: int = 500,
):
    """
    Scrapes schemes data from India Government Schemes portal using Playwright.

    Steps followed:
    1. Enter the portal and wait for the site to load.
    2. Search the schemes container by id 'schemeList', extracting topic schema names and URLs.
    3. Enter into each link one by one and wait for the page to load.
    4. Scrape the data title from id 'schemeTitleLink' and its description from the <p> tag underneath.
    5. Repeat for all remaining topics in the list.
    6. Save all items into formatted file(s) with each Schema as a topic and sub-schemas listed underneath.
    """
    logger.info("Step 1: Launching visible Playwright browser (Chrome) and entering portal...")

    scraped_data = []

    with sync_playwright() as p:
        # Launch visible chromium with anti-automation flags and slow-motion delay
        browser = p.chromium.launch(
            headless=headless,
            slow_mo=slow_mo,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-setuid-sandbox",
            ],
        )

        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            extra_http_headers={
                "sec-ch-ua": '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"macOS"',
                "accept-language": "en-US,en;q=0.9",
            },
        )

        page = context.new_page()

        try:
            logger.info(f"Navigating to {PORTAL_URL}...")
            response = page.goto(PORTAL_URL, timeout=60000, wait_until="networkidle")
            if response and response.status != 200:
                logger.warning(f"Unexpected status code: {response.status}")

            logger.info(f"Page loaded: {page.title()}")

            # Step 2: Search the Schemes by ID "schemeList"
            logger.info("Step 2: Searching for schemes container by ID 'schemeList'...")
            page.wait_for_selector("#schemeList", state="attached", timeout=15000)

            scheme_list_locator = page.locator("#schemeList")
            link_locators = scheme_list_locator.locator("a").all()

            topics = []
            for link in link_locators:
                topic_name = link.inner_text().strip()
                raw_href = link.get_attribute("href") or ""
                full_url = urljoin(PORTAL_URL, raw_href)
                if topic_name and full_url:
                    topics.append({"topic": topic_name, "url": full_url})

            logger.info(f"Found {len(topics)} topics under #schemeList:")
            for idx, t in enumerate(topics, 1):
                logger.info(f"  {idx}. {t['topic']} -> {t['url']}")

            # Step 3, 4, 5: Enter each link one by one and extract schemes
            logger.info("Step 3 & 4 & 5: Iterating through each topic link...")
            for idx, item in enumerate(topics, 1):
                topic_name = item["topic"]
                topic_url = item["url"]
                logger.info(f"\n[{idx}/{len(topics)}] Loading topic: '{topic_name}'...")

                topic_record = {
                    "topic": topic_name,
                    "url": topic_url,
                    "sub_schemes": [],
                }

                try:
                    page.goto(topic_url, timeout=60000, wait_until="networkidle")

                    page_num = 1
                    seen_titles = set()

                    while True:
                        logger.info(f"  Scraping page {page_num} for topic '{topic_name}'...")

                        # Wait for scheme titles or empty state
                        has_schemes = False
                        try:
                            page.wait_for_selector("#schemeTitleLink", state="attached", timeout=8000)
                            has_schemes = True
                        except Exception:
                            if page_num == 1:
                                logger.info(f"  No '#schemeTitleLink' elements found for '{topic_name}'.")

                        if not has_schemes:
                            break

                        scheme_links = page.locator("#schemeTitleLink").all()
                        logger.info(f"  Found {len(scheme_links)} sub-schemes on page {page_num} for '{topic_name}'.")

                        page_first_title = ""
                        for scheme_elem in scheme_links:
                            title = scheme_elem.inner_text().strip()
                            link = scheme_elem.get_attribute("href") or ""
                            if not title:
                                continue
                            if not page_first_title:
                                page_first_title = title

                            # Avoid duplicate additions if page re-renders
                            item_key = (title, link)
                            if item_key in seen_titles:
                                continue
                            seen_titles.add(item_key)

                            # Extract description from the <p> tag under the schemeTitleLink container
                            description = scheme_elem.evaluate(
                                """el => {
                                    // el is the <a> tag inside <h2> inside a card container
                                    let container = el.closest(".search_schemeSectionData__3AS3m");
                                    if (!container) {
                                        // Traverse up to 3 parents to locate card container
                                        let node = el;
                                        for (let i = 0; i < 3; i++) {
                                            if (node && node.parentElement) node = node.parentElement;
                                        }
                                        container = node;
                                    }
                                    if (container) {
                                        let p = container.querySelector("p");
                                        if (p) return p.innerText.trim();
                                    }
                                    return "";
                                }"""
                            )

                            sub_idx = len(topic_record["sub_schemes"]) + 1
                            sub_scheme = {
                                "index": sub_idx,
                                "title": title,
                                "link": link,
                                "description": description,
                            }
                            topic_record["sub_schemes"].append(sub_scheme)
                            logger.info(f"    - Sub-Scheme {sub_idx}: {title}")

                        # Scroll to the end of page to reach pagination controls
                        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        page.wait_for_timeout(1000)

                        # Check the aria-label="Next Page" button
                        next_btn = page.locator('button[aria-label="Next Page"], [aria-label="Next Page"]').first
                        if next_btn.count() == 0:
                            logger.info(f"  No 'Next Page' button found. Ending pagination for '{topic_name}'.")
                            break

                        # Check if Next Page button is disabled
                        is_disabled = (
                            next_btn.is_disabled()
                            or "p-disabled" in (next_btn.get_attribute("class") or "")
                            or next_btn.get_attribute("disabled") is not None
                            or next_btn.get_attribute("aria-disabled") == "true"
                        )

                        if is_disabled:
                            logger.info(
                                f"  'Next Page' button is disabled. Reached end of pagination (page {page_num}) for '{topic_name}'."
                            )
                            break

                        # If enabled, click the button and start scraping the next page
                        logger.info(f"  'Next Page' is enabled. Clicking to load page {page_num + 1}...")
                        next_btn.click()
                        page.wait_for_load_state("networkidle")

                        # Wait for the next page's items to update
                        try:
                            page.wait_for_function(
                                """(prevTitle) => {
                                    let el = document.querySelector("#schemeTitleLink");
                                    return el && el.innerText.trim() !== prevTitle;
                                }""",
                                arg=page_first_title,
                                timeout=8000,
                            )
                        except Exception:
                            page.wait_for_timeout(2000)

                        page_num += 1

                except Exception as ex:
                    logger.error(f"  Error loading topic '{topic_name}' ({topic_url}): {ex}")

                scraped_data.append(topic_record)

        finally:
            browser.close()
            logger.info("Browser closed.")

    # Step 6: Write all items to file named "schema"
    write_output_files(scraped_data, output_base_path)
    return scraped_data


def write_output_files(scraped_data: list, output_base_path: str = DEFAULT_OUTPUT_PATH):
    """
    Writes the scraped data to files.
    Formats with Schema as topic and differentiated sub-schemas underneath.
    Outputs:
    1. <output_base_path> (exact name requested by user: 'schema')
    2. <output_base_path>.txt (readable text format)
    3. <output_base_path>.json (structured JSON format)
    """
    out_dir = os.path.dirname(os.path.abspath(output_base_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total_topics = len(scraped_data)
    total_sub_schemes = sum(len(t.get("sub_schemes", [])) for t in scraped_data)

    lines = []
    lines.append("=" * 80)
    lines.append("INDIA GOVERNMENT SCHEMES DIRECTORY")
    lines.append(f"Total Topics (Schemas): {total_topics}")
    lines.append(f"Total Sub-Schemes: {total_sub_schemes}")
    lines.append("=" * 80)
    lines.append("")

    for topic_item in scraped_data:
        topic = topic_item["topic"]
        url = topic_item["url"]
        sub_schemes = topic_item.get("sub_schemes", [])

        lines.append("=" * 80)
        lines.append(f"SCHEMA (TOPIC): {topic}")
        lines.append(f"Source URL: {url}")
        lines.append(f"Sub-Schemes Available: {len(sub_schemes)}")
        lines.append("=" * 80)

        if not sub_schemes:
            lines.append("  [No sub-schemes currently listed under this category]\n")
            continue

        for sub in sub_schemes:
            s_idx = sub.get("index", "")
            title = sub.get("title", "N/A")
            link = sub.get("link", "N/A")
            desc = sub.get("description", "No description available.")

            lines.append(f"  --- [Sub-Schema {s_idx}] ---")
            lines.append(f"  Title: {title}")
            lines.append(f"  Link:  {link}")
            lines.append(f"  Description:\n    {desc}")
            lines.append("  " + "-" * 76)
        lines.append("")

    text_content = "\n".join(lines)

    # Determine files to write
    files_to_write = [
        output_base_path,
        f"{output_base_path}.txt",
    ]

    for file_path in files_to_write:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        logger.info(f"Successfully saved text output to: {os.path.abspath(file_path)}")

    # Also save structured JSON
    json_path = f"{output_base_path}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scraped_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Successfully saved JSON output to: {os.path.abspath(json_path)}")


if __name__ == "__main__":
    # By default, Chrome will open visibly on screen with a 500ms slow_mo delay
    is_headless = "--headless" in sys.argv
    cli_args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    output_filename = cli_args[0] if cli_args else DEFAULT_OUTPUT_PATH

    scrape_schemes(
        output_base_path=output_filename,
        headless=is_headless,
        slow_mo=0 if is_headless else 500,
    )
