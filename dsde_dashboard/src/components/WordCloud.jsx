'use client';

import * as d3 from 'd3';
import d3Cloud from 'd3-cloud'; // Import the d3-cloud library

export default function WordCloud(
    wordsArray,
    {
        size = (group) => group.size, // Given a grouping of words, returns the size factor for that word
        word = (d) => d.text, // Given an item of the data array, returns the word
        marginTop = 0, // top margin, in pixels
        marginRight = 0, // right margin, in pixels
        marginBottom = 0, // bottom margin, in pixels
        marginLeft = 0, // left margin, in pixels
        width = 640, // outer width, in pixels
        height = 400, // outer height, in pixels
        maxWords = 250, // maximum number of words to extract
        fontFamily = 'sans-serif', // font family
        fontScale = 15, // base font size
        fill = null, // text color, can be a constant or a function of the word
        padding = 0, // amount of padding between the words (in pixels)
        rotate = 0, // a constant or function to rotate the words
        invalidation, // when this promise resolves, stop the simulation
    } = {}
) {
    // The input 'wordsArray' should already be in the format of [['word', 2], ['hi', 5], ...]
    const data = wordsArray
        .sort(([, a], [, b]) => d3.descending(a, b)) // Sort by frequency/size
        .slice(0, maxWords)
        .map(([text, size]) => ({ text, size }));

    const svg = d3
        .create('svg')
        .attr('viewBox', [0, 0, width, height])
        .attr('width', width)
        .attr('font-family', fontFamily)
        .attr('text-anchor', 'middle')
        .attr('style', 'max-width: 100%; height: auto; height: intrinsic;');

    const g = svg
        .append('g')
        .attr('transform', `translate(${marginLeft},${marginTop})`);

    const cloud = d3Cloud()
        .size([
            width - marginLeft - marginRight,
            height - marginTop - marginBottom,
        ])
        .words(data)
        .padding(padding)
        .rotate(rotate)
        .font(fontFamily)
        .fontSize((d) => Math.sqrt(d.size) * fontScale)
        .on('word', ({ size, x, y, rotate, text }) => {
            g.append('text')
                .datum(text)
                .attr('font-size', size)
                .attr('fill', fill)
                .attr('transform', `translate(${x},${y}) rotate(${rotate})`)
                .text(text);
        });

    cloud.start();
    invalidation && invalidation.then(() => cloud.stop());
    return svg.node();
}
