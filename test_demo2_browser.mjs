import { chromium } from 'playwright';

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  
  page.on('console', msg => {
    if (msg.type() === 'error') {
       console.log('BROWSER ERROR:', msg.text());
    } else {
       console.log('BROWSER LOG:', msg.text());
    }
  });

  page.on('pageerror', err => {
    console.error('PAGE ERROR:', err);
  });

  console.log("Navigating to http://localhost:5177/demo2");
  await page.goto('http://localhost:5177/demo2', { waitUntil: 'networkidle' });
  console.log("Page loaded. Waiting 3 seconds...");
  await page.waitForTimeout(3000);
  console.log("Done.");
  await browser.close();
})();
