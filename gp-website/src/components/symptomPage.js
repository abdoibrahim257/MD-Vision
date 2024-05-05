import NavBar from "./navbar";
import '../styles/heroSection.css';
import '../styles/ChatBot.css';
import React from "react";
import { WindupChildren } from "windups";
import SearchBar from "./SearchBar.js";

import robot from '../assets/loveBot.svg';
import SearchBarList from "./SearchBarList.js";

const SymptomPage = () => {

    const [results, setResults] = React.useState([])

    return (
        <div>
            {/* an nave bar will be here */}
            <NavBar />

            <div className="content">
                <div className="textContainer">
                    <img className="bot" src={robot} alt="Maven bot" />
                    <div>
                        <div className="welcome">
                            <p>👋 Hello there! I'm Maven, your friendly health assistant here to lend a helping hand!</p>
                        </div>
                        <WindupChildren>
                            <p className="help">Welcome to our well-being hub! Unsure of your symptoms? No worries, pick or search, and let's navigate together towards your best self! 🌟</p>
                            {/* <Pace ms={5}><p className="help">Welcome to our little corner of the web where we're all about your well-being. If you're here, chances are you're looking for some guidance on how to feel your best. Well, you've come to the right place!</p></Pace> */}
                            {/* <Pace ms={5}><p className="help">But first things first, let me give you a quick rundown. Below, you'll find a list of common symptoms. Don't worry if you're not sure which one matches what you're feeling—just give it your best shot, or if you prefer, you can search for your symptoms directly. Once you've selected or searched for your symptom, I'll do my best to provide you with helpful information and guidance.</p></Pace>
                            <Pace ms={5}><p className="help">So, take a deep breath, relax, and let's get started on this journey towards feeling your absolute best! 💫.</p></Pace>            */}
                        </WindupChildren>
                    </div>
                </div>
                <SearchBar setResults  =  {setResults}/>
                <SearchBarList results = {results} />

            </div>
        </div>
    );
}

export default SymptomPage;