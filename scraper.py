from requests_html import HTMLSession
from time import sleep
import csv


def scrape(url, name):
    hs = HTMLSession()

    try:
        url = url.replace("dp", "product-reviews")
    except Exception as e:
        print(e)
        quit()

    r = hs.get(url=url, headers={
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7,zh-TW;q=0.6',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'none',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36'})

    comments = r.html.find('div.a-section.review.aok-relative')

    fw = open(f'{name}.csv', 'a', encoding='utf8')  # output file
    writer = csv.writer(fw, lineterminator='\n')

    for a in comments:

        comment, star = 'NA', 'NA'  # initialize critic and text

        commentChunk = a.find('span.a-size-base.review-text.review-text-content > span')
        if commentChunk:
            comment = commentChunk[0].text.strip()

        starChunk = a.find('i > span.a-icon-alt')
        if starChunk: star = starChunk[0].text.strip()

        # star = a.find('i > span.a-icon-alt')[0].text
        # comment = a.find('span.a-size-base.review-text.review-text-content > span')[0].text

        writer.writerow([comment, star])

    fw.close()
    sleep(.75)
    pagination(r, name)

    r.close()


def pagination(attempt, name):
    next_page = attempt.html.find('li.a-last > a')
    if next_page:
        new_url = ''.join(next_page[0].absolute_links)
        # print(new_url)
        scrape(new_url, name)