import React, { Component } from "react";

//Components
import Chatbox from "./Chatbox/Chatbox";


class App extends Component {
    
    
    //The "HTML" that is rendered on screen
    render() {
        return (
            <div>
                <Chatbox />
            </div>
        );
    }

}

export default App; 