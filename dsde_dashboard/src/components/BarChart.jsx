import React from 'react';
import dynamic from 'next/dynamic';

// Dynamically import the Plot component with ssr: false to prevent SSR
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

const BarChart = ({ data }) => {
    if (!data) return null;

    // Extract the categories and values from the data
    const categories = data.map((d) => d[0]);
    const values = data.map((d) => d[1]);

    return (
        <Plot
            data={[
                {
                    type: 'bar',
                    x: categories,
                    y: values,
                    marker: {
                        color: [
                            '#9b5de5',
                            '#f15bb5',
                            '#fee440',
                            '#00bbf9',
                            '#00f5d4',
                        ],
                    },
                },
            ]}
            layout={{
                xaxis: {
                    title: 'Keywords',
                },
                yaxis: {
                    title: 'Relevance Score',
                },
                margin: {
                    t: 0,
                    r: 30,
                    b: 40,
                    l: 80,
                },
            }}
        />
    );
};

export default BarChart;
