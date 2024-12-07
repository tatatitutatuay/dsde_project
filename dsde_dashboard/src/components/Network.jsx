'use client';

import React, { useEffect, useState, useRef } from 'react';
import { Network, DataSet } from 'vis-network/standalone'; // Import DataSet correctly

export default function NetworkComponent({ path }) {
    const networkContainerRef = useRef(null);
    const [nodes, setNodes] = useState(null);
    const [edges, setEdges] = useState(null);

    // Fetch the data from the provided path
    useEffect(() => {
        fetch(path) // Update with your actual JSON path
            .then((response) => response.json())
            .then((data) => {
                console.log(data); // Debugging: check the fetched data

                // Initialize vis.DataSet for nodes and edges
                const nodesDataSet = new DataSet(data.nodes);
                const edgesDataSet = new DataSet(data.edges);

                setNodes(nodesDataSet);
                setEdges(edgesDataSet);
            });
    }, [path]);

    // Initialize the network after nodes and edges are loaded
    useEffect(() => {
        if (nodes && edges) {
            // Ensure nodes and edges are not null
            const options = {
                nodes: {
                    shape: 'box',
                },
                edges: {
                    color: '#D3D3D3',
                    width: 0.5,
                    smooth: true,
                },
                physics: {
                    enabled: true,
                },
                interaction: {
                    dragNodes: true,
                    dragView: true,
                    tooltipDelay: 200,
                },
            };

            // Create the network visualization
            const network = new Network(
                networkContainerRef.current,
                { nodes, edges },
                options
            );

            // Event listener for node clicks
            network.on('click', function (params) {
                // Reset all node colors
                nodes.forEach((node) => {
                    nodes.update({ id: node.id, color: '#97C2FC' }); // Default color
                });

                if (params.nodes.length > 0) {
                    const clickedNodeId = params.nodes[0]; // Get the clicked node ID

                    // Highlight the clicked node
                    nodes.update({ id: clickedNodeId, color: '#FF5733' }); // Clicked node color

                    // Find all edges connected to the clicked node
                    const connectedEdges = edges.get({
                        filter: (edge) =>
                            edge.from === clickedNodeId ||
                            edge.to === clickedNodeId,
                    });

                    // Highlight connected nodes
                    connectedEdges.forEach((edge) => {
                        const connectedNodeId =
                            edge.from === clickedNodeId ? edge.to : edge.from;

                        // Update the connected node color
                        nodes.update({ id: connectedNodeId, color: '#FFC300' }); // Highlight color
                    });
                }
            });
        }
    }, [nodes, edges]); // Re-run when nodes or edges change

    return (
        <div
            ref={networkContainerRef}
            style={{
                height: '750px',
                width: '100%',
                border: '1px solid black',
            }}
        />
    );
}
