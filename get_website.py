import re
import requests
from bs4 import BeautifulSoup


def get_html(website_link, result, seen, max_depth):
    """
    get all text content and all sub-links from a website
    :param max_depth: depth to be scraped
    :param seen: links already been seen
    :param result: string to restore text from the website_link
    :param website_link: current page
    :return: text without html label
    """
    if max_depth == 0:
        return result  # stop scraping if reach the max depth
    max_depth -= 1

    if website_link not in seen:  # prevent repeated scraping of the same link
        seen.append(website_link)

        if website_link:
            try:
                header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'}
                response = requests.get(url=website_link, headers=header)

                response.encoding = response.apparent_encoding  # give the response object a code
                # response.raise_for_status()  # check if an error occur
                html = response.text  # assign the text of the response to html

                soup = BeautifulSoup(html, 'html.parser')  # parse text in html with beautifulsoup
                result += re.sub("(\<.*?\>)", "", soup.text).replace('\n', '')  # remove all html label by re

                tag_a = soup.find_all('a')  # find tags start with 'a'
                hyper_links = []  # restore all links appear on this web page
                for tag in tag_a:
                    link = tag.get('href')  # hyperlinks usually start with 'href='
                    if not link:
                        continue
                    website_link_prefix = website_link[:-1] if website_link[-1] == '/' else website_link
                    hyper_links.append(website_link_prefix + link)

                hyper_links = list(set(hyper_links))  # remove all duplicated links

                if hyper_links:
                    for link in hyper_links:
                        if link != website_link:
                            get_html(link, result=result, seen=seen, max_depth=max_depth)

                return result

            except Exception as e:
                # print(e)
                return result

        else:
            return result


def search_products(website_link, max_depth=2):
    """
    scape text from a given website
    :param max_depth: default as 2
    :param website_links: url
    :return: BeautifulSoup Object
    """
    # start = time()
    # print('start scraping...')

    text = ' '
    result = get_html(website_link, text, [], max_depth)
    result = ' '.join(result.split())
    result = re.sub("(\<.*?\>)", "", result)
    # comp = re.compile('[^A-Z^a-z^0-9^]')  # remove all special characters
    # result = comp.sub(' ', result)
    # try:
    #     result = BeautifulSoup(result, 'lxml')
    # except Exception:
    #     pass

    # fw = open(f'{file_name}.csv', 'a', encoding='utf8')
    # writer = csv.writer(fw, lineterminator='\n')
    # writer.writerow([website_links[p], result])

    # print(f'scraping finished, run time: {time() - start}s.')

    return result
