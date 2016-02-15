# -*- coding: utf-8 -*-
import scrapy


from quote.items import QuoteItem


class QuoteSpider(scrapy.Spider):
    name = "quote"
    allowed_domains = ["mingyan.xyzdict.com"]
    start_urls = [
        "http://mingyan.xyzdict.com/mingren/"
    ]

    def parse(self, response):
        for url in response.css('ul li a::attr("href")').re('.*/mingren/.*'):
            yield scrapy.Request(response.urljoin(url), self.parse_page)
        # for href in response.xpath('//ul/li/a/@href'):
        #     url = response.urljoin(href.extract()
        #     yield scrapy.Request(url, callback=self.parse_page)

    def parse_page(self, response):
        author = response.xpath('//span/text()').extract()[0]
        for sel in response.xpath('//ul/li'):
            quote = QuoteItem()
            quote['w'] = sel.xpath('a/text()').extract()
            if not quote['w']:
                continue
            quote['w'] = quote['w'][0]
            quote['a'] = author
            yield quote
