import React from 'react'
import { FaSearch } from 'react-icons/fa'
import '../styles/SearchBar.css'

const SearchBar = ( { setResults } ) => {
    const [input, setInput] = React.useState('')

    const fetchData = (data) => {
        fetch("https://shad-honest-anchovy.ngrok-free.app/maven", {
            headers: new Headers({
                "ngrok-skip-browser-warning": "69420",
            }),
        })
        .then(response => response.json())
        .then(json => {
            const symptoms = json.fileList
            const results = symptoms.filter((symptom) => {
                return data && (symptom.toLowerCase().includes(data.toLowerCase())) 
            })
            // console.log(results)
            setResults(results)
        });
    }

    const handleSearch = (data) => {
        setInput(data)
        fetchData(data)
    }

    
    return (
        <div className='input-wrapper' tabIndex="-1">
            <FaSearch id="search-icon" />
            <input type="text" id="search" placeholder="Search for your symptoms"
                value={input} onChange={(e) => handleSearch(e.target.value)} />
        </div>
    )
}

export default SearchBar