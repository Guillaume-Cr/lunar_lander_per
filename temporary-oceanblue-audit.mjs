import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

const base = (process.env.PREVIEW_BASE || '').replace(/\/$/, '');
const expectedVersion = process.env.ARTIST_VERSION || '2026-08-03-8';
if (!base) throw new Error('PREVIEW_BASE is required');

const output = 'oceanblue-v7-audit';
fs.rmSync(output, { recursive: true, force: true });
for (const folder of ['full', 'viewport', 'hero']) {
  for (const profile of ['desktop', 'tablet', 'mobile']) {
    fs.mkdirSync(path.join(output, folder, profile), { recursive: true });
  }
}

const profiles = [
  { name: 'desktop', width: 1440, height: 960, quality: 68 },
  { name: 'tablet', width: 1024, height: 900, quality: 66 },
  { name: 'mobile', width: 390, height: 844, quality: 64 },
];

const slug = (route) =>
  route === '/' ? 'home' : route.replace(/^\//, '').replace(/\//g, '__');

async function waitForVersion(browser) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const deadline = Date.now() + 15 * 60 * 1000;
  let last = '';

  while (Date.now() < deadline) {
    try {
      await page.goto(`${base}/services/depression`, {
        waitUntil: 'domcontentloaded',
        timeout: 60_000,
      });
      await page.waitForTimeout(1500);
      const state = await page.evaluate(() => ({
        ready: document.body.getAttribute('data-artist-sitewide-v7'),
        version: document.body.getAttribute('data-artist-sitewide-v7-version'),
        titleClearance: document.body.getAttribute('data-v7-title-clearance'),
      }));
      last = JSON.stringify(state);
      if (state.ready === 'true' && state.version === expectedVersion) {
        await page.close();
        return;
      }
    } catch (error) {
      last = String(error);
    }
    await page.waitForTimeout(12_000);
  }

  await page.close();
  throw new Error(`Preview did not reach artist version ${expectedVersion}. Last state: ${last}`);
}

async function discoverRoutes(page) {
  const response = await page.goto(`${base}/sitemap.xml`, {
    waitUntil: 'domcontentloaded',
    timeout: 60_000,
  });
  if (!response || response.status() >= 400) {
    throw new Error(`Unable to load sitemap: HTTP ${response?.status()}`);
  }
  const source = await page.locator('body').innerText();
  const routes = [...source.matchAll(/https:\/\/oceanbluetherapy\.ca([^\s<]*)/g)]
    .map((match) => match[1] || '/')
    .map((route) => route.replace(/\?.*$/, '') || '/');
  return [...new Set(routes)];
}

async function prepare(page) {
  await page.evaluate(async () => {
    const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
    const step = Math.max(460, Math.floor(innerHeight * 0.82));
    for (let y = 0; y < document.documentElement.scrollHeight; y += step) {
      scrollTo(0, y);
      await pause(35);
    }
    scrollTo(0, 0);

    document.querySelectorAll('iframe').forEach((frame) => {
      const box = frame.getBoundingClientRect();
      const label = `${frame.src || ''} ${frame.title || ''}`;
      const style = getComputedStyle(frame);
      if (
        /netlify|drawer|collaborate/i.test(label) ||
        (style.position === 'fixed' &&
          box.bottom >= innerHeight - 4 &&
          box.height < 180)
      ) {
        frame.style.setProperty('display', 'none', 'important');
      }
    });
  });

  await page.addStyleTag({
    content: `
      [data-netlify-drawer], netlify-drawer, #netlify-drawer,
      iframe[src*="netlify"], iframe[title*="Netlify"] { display: none !important; }
      body.audit-hero-capture #header,
      body.audit-hero-capture .top-header,
      body.audit-hero-capture #top-header,
      body.audit-hero-capture .top-bar,
      body.audit-hero-capture .mobile-top-bar { visibility: hidden !important; }
    `,
  });
  await page.waitForTimeout(500);
}

async function inspect(page, route, profile) {
  return page.evaluate(
    ({ route, profile, expectedVersion }) => {
      const visible = (node) => {
        if (!node) return false;
        const style = getComputedStyle(node);
        const box = node.getBoundingClientRect();
        return (
          style.display !== 'none' &&
          style.visibility !== 'hidden' &&
          Number(style.opacity) !== 0 &&
          box.width > 2 &&
          box.height > 2
        );
      };

      const rect = (node) => {
        if (!node) return null;
        const box = node.getBoundingClientRect();
        return {
          top: box.top,
          right: box.right,
          bottom: box.bottom,
          left: box.left,
          width: box.width,
          height: box.height,
        };
      };

      const hero = document.querySelector(
        'article#main > #side-page-banner, article#main > .side-page-title, article#main > #banner, #banner',
      );
      const titleCandidates = [
        ...document.querySelectorAll(
          'article#main > #side-page-banner h1, article#main > .side-page-title h1, article#main > #banner h1, article#main .page-primary-title, article#main h1, article#main > #side-page-banner h2, article#main > .side-page-title h2',
        ),
      ];
      const title = titleCandidates.find(visible) || null;
      const inner = hero?.querySelector(':scope > .inner') || null;
      const image = hero?.querySelector(':scope > img') || null;
      const innerBox = rect(inner);
      const titleBox = rect(title);
      const imageBox = rect(image);
      let heroIssue = '';

      if (innerBox && titleBox && imageBox) {
        const horizontal = imageBox.left > innerBox.left + innerBox.width * 0.55;
        if (horizontal && titleBox.right > imageBox.left + 3) {
          heroIssue = 'title crosses into hero image';
        }
        if (
          !horizontal &&
          profile === 'mobile' &&
          innerBox.bottom > imageBox.top + 3
        ) {
          heroIssue = 'stacked hero text overlaps image';
        }
      }

      const headerNodes = [
        document.querySelector('#top-header'),
        document.querySelector('.top-header'),
        document.querySelector('.top-bar'),
        document.querySelector('#header'),
        document.querySelector('site-header'),
      ].filter(visible);
      const headerBottom = headerNodes.reduce((maximum, node) => {
        const box = node.getBoundingClientRect();
        if (box.bottom <= 0 || box.top > 10) return maximum;
        return Math.max(maximum, box.bottom);
      }, 0);
      const titleHiddenByHeader = Boolean(
        titleBox &&
          headerBottom &&
          titleBox.bottom > 0 &&
          titleBox.top < headerBottom + 8,
      );

      const journey = document.querySelector('.home-therapy-journey');
      const journeyNext = journey?.nextElementSibling || null;
      const journeyImageCard =
        journey?.querySelector('.section-image.style-right.vertical') || null;
      const journeySteps = [
        ...document.querySelectorAll('.home-therapy-journey .therapy-step'),
      ];
      const journeyBox = rect(journey);
      const nextBox = rect(journeyNext);
      const imageCardBox = rect(journeyImageCard);
      const lastStepBox = rect(journeySteps[journeySteps.length - 1]);
      const journeyOverlap =
        journeyBox && nextBox ? Math.max(0, journeyBox.bottom - nextBox.top) : 0;
      const journeyStackGap =
        profile === 'mobile' && imageCardBox && lastStepBox
          ? imageCardBox.top - lastStepBox.bottom
          : null;
      const stepOffsets = journeySteps.map((step) =>
        Math.round(step.getBoundingClientRect().left),
      );

      const contactBox = rect(
        document.querySelector('.side-page-doyou.contact-form'),
      );
      const mobileCta = document.querySelector('.mobile-cta-banner');
      const h1s = [...document.querySelectorAll('h1')];

      const clipped = [
        ...document.querySelectorAll('a, button, input, textarea, select'),
      ]
        .filter(visible)
        .filter((node) => !node.closest('#navPanel'))
        .filter((node) => {
          const box = node.getBoundingClientRect();
          return box.left < -2 || box.right > innerWidth + 2;
        })
        .map((node) =>
          node.textContent.replace(/\s+/g, ' ').trim().slice(0, 80),
        );

      const brokenImages = [...document.images]
        .filter(visible)
        .filter((node) => node.complete && node.naturalWidth === 0)
        .filter((node) => {
          try {
            return (
              new URL(node.currentSrc || node.src, location.href).origin ===
              location.origin
            );
          } catch {
            return true;
          }
        })
        .map((node) => node.currentSrc || node.src || node.alt || 'unknown image');

      return {
        route,
        profile,
        version: document.body.getAttribute('data-artist-sitewide-v7-version'),
        overflow: document.documentElement.scrollWidth - innerWidth,
        h1Count: h1s.length,
        visualTitle: title?.textContent.replace(/\s+/g, ' ').trim() || '',
        heroIssue,
        titleHiddenByHeader,
        titleClearanceApplied:
          document.body.getAttribute('data-v7-title-clearance') === 'true',
        clipped,
        brokenImages,
        mobileCtaVisible: profile === 'mobile' && visible(mobileCta),
        journeyOverlap,
        journeyStackGap,
        journeyImageLoaded: journeyImageCard
          ? Boolean(journeyImageCard.querySelector('img')?.naturalWidth)
          : null,
        stepOffsets,
        contactWidth: contactBox?.width || null,
        contactHeight: contactBox?.height || null,
      };
    },
    { route, profile, expectedVersion },
  );
}

async function captureHero(page, route, profile) {
  const selectors = [
    'article#main > #side-page-banner',
    'article#main > .side-page-title',
    'article#main > #banner',
    '#banner',
    'article#main > :first-child',
  ];
  let locator = null;
  for (const selector of selectors) {
    const candidate = page.locator(selector).first();
    if ((await candidate.count()) && (await candidate.isVisible())) {
      locator = candidate;
      break;
    }
  }
  if (!locator) return;

  await page.evaluate(() => document.body.classList.add('audit-hero-capture'));
  await locator.screenshot({
    path: path.join(output, 'hero', profile, `${slug(route)}.jpg`),
    type: 'jpeg',
    quality: 74,
  });
  await page.evaluate(() => document.body.classList.remove('audit-hero-capture'));
}

const browser = await chromium.launch({ headless: true });
await waitForVersion(browser);
const discoveryPage = await browser.newPage();
const routes = await discoverRoutes(discoveryPage);
await discoveryPage.close();
const report = [];

for (const route of routes) {
  for (const profile of profiles) {
    const context = await browser.newContext({
      viewport: { width: profile.width, height: profile.height },
      deviceScaleFactor: 1,
    });
    const page = await context.newPage();
    const errors = [];

    page.on('pageerror', (error) => errors.push(`pageerror: ${error.message}`));
    page.on('requestfailed', (request) => {
      try {
        const url = new URL(request.url());
        if (url.origin === new URL(base).origin) {
          errors.push(
            `requestfailed: ${url.pathname}: ${request.failure()?.errorText || 'unknown'}`,
          );
        }
      } catch {}
    });

    try {
      const response = await page.goto(`${base}${route}`, {
        waitUntil: 'networkidle',
        timeout: 90_000,
      });
      await page.waitForTimeout(650);
      await prepare(page);
      const metrics = await inspect(page, route, profile.name);

      await page.screenshot({
        path: path.join(output, 'viewport', profile.name, `${slug(route)}.jpg`),
        type: 'jpeg',
        quality: profile.quality,
      });
      await page.screenshot({
        path: path.join(output, 'full', profile.name, `${slug(route)}.jpg`),
        type: 'jpeg',
        quality: profile.quality,
        fullPage: true,
      });
      await captureHero(page, route, profile.name);

      report.push({ ...metrics, status: response?.status() || null, errors });
      console.log(`${profile.name.padEnd(7)} ${route}`);
    } catch (error) {
      report.push({
        route,
        profile: profile.name,
        failed: true,
        errors: [...errors, String(error)],
      });
      console.error(`FAILED ${profile.name} ${route}: ${error}`);
    } finally {
      await context.close();
    }
  }
}

await browser.close();
fs.writeFileSync(path.join(output, 'metrics.json'), JSON.stringify(report, null, 2));

const problems = report.filter((item) => {
  const journeyNeedsStagger = item.route === '/' && item.stepOffsets?.length === 3;
  const staggerRange = journeyNeedsStagger
    ? Math.max(...item.stepOffsets) - Math.min(...item.stepOffsets)
    : null;
  return (
    item.failed ||
    item.status >= 400 ||
    item.version !== expectedVersion ||
    item.overflow > 2 ||
    item.errors?.length ||
    item.clipped?.length ||
    item.brokenImages?.length ||
    !item.h1Count ||
    item.heroIssue ||
    item.titleHiddenByHeader ||
    item.mobileCtaVisible ||
    item.journeyOverlap > 2 ||
    item.journeyImageLoaded === false ||
    (item.journeyStackGap !== null &&
      (item.journeyStackGap < 16 || item.journeyStackGap > 100)) ||
    (journeyNeedsStagger &&
      staggerRange < (item.profile === 'mobile' ? 6 : 18)) ||
    (item.route === '/contact-us' &&
      item.profile === 'mobile' &&
      item.contactWidth < 340) ||
    (item.route === '/contact-us' &&
      item.contactHeight > (item.profile === 'mobile' ? 1700 : 1050))
  );
});

const summary = [
  '# Ocean Blue V7 Independent Full Visual Audit',
  '',
  `Artist version: ${expectedVersion}`,
  `Routes: ${routes.length}`,
  `Rendered page states: ${report.length}`,
  'Screenshots per state: viewport, full page, and hero or opening section.',
  `Objective rejection findings: ${problems.length}`,
  '',
  ...problems.map((item) => {
    const staggerRange = item.stepOffsets?.length
      ? Math.max(...item.stepOffsets) - Math.min(...item.stepOffsets)
      : null;
    return `- ${item.route} (${item.profile}): ${[
      item.failed ? 'capture failed' : '',
      item.status >= 400 ? `HTTP ${item.status}` : '',
      item.version !== expectedVersion ? `wrong version ${item.version}` : '',
      item.overflow > 2 ? `overflow ${item.overflow}px` : '',
      item.errors?.length ? `${item.errors.length} local browser errors` : '',
      item.clipped?.length ? `${item.clipped.length} clipped controls` : '',
      item.brokenImages?.length
        ? `${item.brokenImages.length} broken local images`
        : '',
      !item.h1Count ? 'no H1 in document' : '',
      item.heroIssue || '',
      item.titleHiddenByHeader
        ? 'visual title remains under the fixed header'
        : '',
      item.mobileCtaVisible ? 'floating mobile CTA obscures content' : '',
      item.journeyOverlap > 2
        ? `journey overlap ${item.journeyOverlap}px`
        : '',
      item.journeyImageLoaded === false ? 'journey image not loaded' : '',
      item.journeyStackGap !== null &&
      (item.journeyStackGap < 16 || item.journeyStackGap > 100)
        ? `journey stack gap ${item.journeyStackGap}px`
        : '',
      item.route === '/' &&
      staggerRange !== null &&
      staggerRange < (item.profile === 'mobile' ? 6 : 18)
        ? 'journey stages are not visibly staggered'
        : '',
      item.route === '/contact-us' &&
      item.profile === 'mobile' &&
      item.contactWidth < 340
        ? `contact width ${item.contactWidth}px`
        : '',
      item.route === '/contact-us' &&
      item.contactHeight > (item.profile === 'mobile' ? 1700 : 1050)
        ? `contact height ${item.contactHeight}px`
        : '',
    ]
      .filter(Boolean)
      .join(', ')}`;
  }),
  '',
  'A zero objective count is required before final visual sign-off.',
].join('\n');

fs.writeFileSync(path.join(output, 'summary.md'), summary);
if (problems.length) {
  throw new Error(`Independent full visual audit found ${problems.length} issue(s).`);
}
