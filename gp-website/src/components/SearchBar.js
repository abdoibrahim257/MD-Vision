import React, {useRef} from 'react'
import { FaSearch } from 'react-icons/fa'
import '../styles/SearchBar.css'
import { json } from 'react-router-dom'

const SearchBar = ( { setResults } ) => {
    const [input, setInput] = React.useState('')
    // const wrapperRef = useRef(null)

    const fetchData = (data) => {
        fetch("https://jsonplaceholder.typicode.com/users")
            .then((response) => (response.json())) //geeting response and converting it to json
            .then((json) => {
                const results = json.filter((user) => {
                    return data && user && user.name && user.name.toLowerCase().includes(data.toLowerCase());
                }); //this must be done on the backend side just for prototype
                // console.log(results);
                setResults(results);
            });
    }

    const handleSearch = (data) => {
        setInput(data)
        fetchData(data)
    }

    // const handleFocus = () => {
    //     if (wrapperRef.current) {
    //         wrapperRef.current.focus();
    //     }
    // }
    
    return (
        <div className='input-wrapper' tabindex="-1">
            <FaSearch id="search-icon" />
            <input type="text" id="search" placeholder="Search for your symptoms"
                value={input} onChange={(e) => handleSearch(e.target.value)} />
        </div>
    )
}

export default SearchBar